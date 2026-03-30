"""Compare Jacobian-based gradient computation vs the existing per-feature loop.

Verifies numerical equivalence and benchmarks wall-clock time.

Usage:
    PYTHONPATH=. python tests/test_jacobian_vs_loop.py
"""

import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feature_circuit_discovery.data import set_model

set_model("gemma-2-2b")

from experiments.feature_grad_exp_optimized import (
    compute_gradient_matrix,
    compute_gradient_matrix_jacobian,
    compute_gradient_matrices_batch,
    load_sae,
)
from feature_circuit_discovery.data import MODEL_ID, canonical_sae_filenames


def main():
    # Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device)

    prompt = "Is 500>300? "
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)

    upstream_layer = 0
    downstream_layer = 10

    # Find actually-active features for a meaningful test
    last_pos = inputs.shape[1] - 1
    sae0 = load_sae(canonical_sae_filenames[upstream_layer], device)
    sae1 = load_sae(canonical_sae_filenames[downstream_layer], device)
    with torch.no_grad():
        outputs = model(inputs, output_hidden_states=True)
        acts0 = sae0.encode(outputs.hidden_states[upstream_layer + 1].float())
        acts1 = sae1.encode(outputs.hidden_states[downstream_layer + 1].float())
        up_feats = torch.where(acts0[0, last_pos, :] > 0)[0][:30]
        down_feats = torch.where(acts1[0, last_pos, :] > 0)[0][:30]
        del outputs
    gc.collect()

    m, n = len(up_feats), len(down_feats)
    print(f"Upstream features (m={m}): {up_feats.tolist()}")
    print(f"Downstream features (n={n}): {down_feats.tolist()}")
    print()

    # --- Warmup (first run loads SAEs, JIT compiles, etc.) ---
    print("Warmup run...")
    _ = compute_gradient_matrix(
        inputs, upstream_layer, downstream_layer,
        up_feats[:3], down_feats[:3], model,
    )
    _ = compute_gradient_matrix_jacobian(
        inputs, upstream_layer, downstream_layer,
        up_feats[:3], down_feats[:3], model, vectorize=True,
    )
    print("Warmup done.\n")
    gc.collect()

    # --- Loop-based (existing) ---
    print("=" * 50)
    print("Loop-based (existing compute_gradient_matrix)")
    print("=" * 50)
    t0 = time.perf_counter()
    grad_loop = compute_gradient_matrix(
        inputs, upstream_layer, downstream_layer,
        up_feats, down_feats, model, verbose=True,
    )
    t_loop = time.perf_counter() - t0
    print(f"Time: {t_loop:.3f}s")
    print(f"Shape: {grad_loop.shape}")
    print(f"Max abs: {grad_loop.abs().max():.6f}")
    print(f"Nonzero: {grad_loop.count_nonzero()}/{grad_loop.numel()}")
    print()

    # --- Jacobian vectorized ---
    print("=" * 50)
    print("Jacobian (vectorize=True)")
    print("=" * 50)
    try:
        t0 = time.perf_counter()
        grad_jac_vec = compute_gradient_matrix_jacobian(
            inputs, upstream_layer, downstream_layer,
            up_feats, down_feats, model, verbose=True, vectorize=True,
        )
        t_jac_vec = time.perf_counter() - t0
        print(f"Time: {t_jac_vec:.3f}s")
        print(f"Shape: {grad_jac_vec.shape}")
        print(f"Max abs: {grad_jac_vec.abs().max():.6f}")

        # Compare
        diff = (grad_loop.float() - grad_jac_vec.float()).abs()
        print(f"Max diff vs loop: {diff.max():.2e}")
        print(f"Mean diff vs loop: {diff.mean():.2e}")
        if diff.max() < 1e-3:
            print("PASS: results match")
        else:
            print("MISMATCH: results differ significantly")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        t_jac_vec = None
    print()

    # --- Jacobian non-vectorized ---
    print("=" * 50)
    print("Jacobian (vectorize=False)")
    print("=" * 50)
    try:
        t0 = time.perf_counter()
        grad_jac_novec = compute_gradient_matrix_jacobian(
            inputs, upstream_layer, downstream_layer,
            up_feats, down_feats, model, verbose=True, vectorize=False,
        )
        t_jac_novec = time.perf_counter() - t0
        print(f"Time: {t_jac_novec:.3f}s")
        print(f"Shape: {grad_jac_novec.shape}")
        print(f"Max abs: {grad_jac_novec.abs().max():.6f}")

        diff = (grad_loop.float() - grad_jac_novec.float()).abs()
        print(f"Max diff vs loop: {diff.max():.2e}")
        print(f"Mean diff vs loop: {diff.mean():.2e}")
        if diff.max() < 1e-3:
            print("PASS: results match")
        else:
            print("MISMATCH: results differ significantly")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        t_jac_novec = None
    print()

    # --- Jacobian with multiple downstream layers ---
    print("=" * 50)
    print("Multi-downstream: Jacobian per layer pair vs batch loop")
    print("=" * 50)

    # Test: upstream=0, downstream=[5, 10, 15]
    down_layers = [5, 10, 15]
    down_feats_multi = {}
    with torch.no_grad():
        outputs = model(inputs, output_hidden_states=True)
        for dl in down_layers:
            sae_dl = load_sae(canonical_sae_filenames[dl], device)
            acts_dl = sae_dl.encode(outputs.hidden_states[dl + 1].float())
            down_feats_multi[dl] = torch.where(acts_dl[0, last_pos, :] > 0)[0][:20]
        del outputs
    gc.collect()

    # Batch loop (existing)
    t0 = time.perf_counter()
    results_batch, _ = compute_gradient_matrices_batch(
        inputs,
        upstream_layer_idx=upstream_layer,
        downstream_pairs=[(dl, down_feats_multi[dl]) for dl in down_layers],
        upstream_features=up_feats,
        model=model,
    )
    t_batch = time.perf_counter() - t0
    print(f"Batch loop (1 fwd, sequential bwd): {t_batch:.3f}s")

    # Jacobian per layer pair — correctness check with vectorize=False
    results_jac = []
    for dl in down_layers:
        mat = compute_gradient_matrix_jacobian(
            inputs, upstream_layer, dl,
            up_feats, down_feats_multi[dl], model, vectorize=False,
        )
        results_jac.append(mat)

    # Verify exact match
    all_match = True
    for i, dl in enumerate(down_layers):
        diff = (results_batch[i].float() - results_jac[i].float()).abs().max()
        print(f"  Layer {dl} max diff: {diff:.2e}", "PASS" if diff < 1e-3 else "MISMATCH")
        if diff >= 1e-3:
            all_match = False

    # Jacobian per layer pair — speed benchmark with vectorize=True
    t0 = time.perf_counter()
    for dl in down_layers:
        _ = compute_gradient_matrix_jacobian(
            inputs, upstream_layer, dl,
            up_feats, down_feats_multi[dl], model, vectorize=True,
        )
    t_jac_multi = time.perf_counter() - t0
    print(f"Batch loop (1 full fwd, seq bwd):    {t_batch:.3f}s")
    print(f"Jacobian per pair (partial fwd+vmap): {t_jac_multi:.3f}s")

    ratio = t_batch / t_jac_multi if t_jac_multi > 0 else float('inf')
    print(f"  -> {ratio:.1f}x {'faster' if t_jac_multi < t_batch else 'slower'}")
    print()

    # --- Summary ---
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Single layer pair (0→{downstream_layer}):")
    print(f"  Loop:                  {t_loop:.3f}s")
    if t_jac_vec is not None:
        print(f"  Jacobian (vectorized): {t_jac_vec:.3f}s  ({t_loop/t_jac_vec:.1f}x {'faster' if t_jac_vec < t_loop else 'slower'})")
    if t_jac_novec is not None:
        print(f"  Jacobian (no vmap):    {t_jac_novec:.3f}s  ({t_loop/t_jac_novec:.1f}x {'faster' if t_jac_novec < t_loop else 'slower'})")
    print(f"Multi-downstream (0→[5,10,15]):")
    print(f"  Batch loop:            {t_batch:.3f}s")
    print(f"  Jacobian per pair:     {t_jac_multi:.3f}s  ({t_batch/t_jac_multi:.1f}x {'faster' if t_jac_multi < t_batch else 'slower'})")


if __name__ == "__main__":
    main()
