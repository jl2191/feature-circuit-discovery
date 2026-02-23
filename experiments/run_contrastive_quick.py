"""Quick contrastive experiment on Gemma 3 1B — scoped to a few layers for fast verification.

Usage:
    poetry run python -m experiments.run_contrastive_quick
"""

import gc
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from experiments.feature_grad_exp_optimized import (
    compute_gradient_matrices_batch,
    get_contrastive_features,
    load_sae,
    _sae_cache,
)
from feature_circuit_discovery.data import MODEL_ID, MAX_SAE_LAYER, canonical_sae_filenames
from feature_circuit_discovery.export import export_circuit_json, export_circuit_html

# ---------------------------------------------------------------------------
# Configuration — scoped down for quick verification
# ---------------------------------------------------------------------------
N_FEATURES = 30          # features per layer
START_LAYER = 8          # start from middle layers (more interesting features)
END_LAYER = 12           # 4 layers: 8, 9, 10, 11

YES_PROMPTS = [
    "Is 847 greater than 231? Answer:",
    "Is 965 greater than 142? Answer:",
    "Is 703 greater than 389? Answer:",
    "Is 556 greater than 214? Answer:",
    "Is 891 greater than 467? Answer:",
]
NO_PROMPTS = [
    "Is 231 greater than 847? Answer:",
    "Is 142 greater than 965? Answer:",
    "Is 389 greater than 703? Answer:",
    "Is 214 greater than 556? Answer:",
    "Is 467 greater than 891? Answer:",
]

if __name__ == "__main__":
    # Device selection: MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32
        ).to(device)
        dtype_str = "float32"
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
        dtype_str = "float32"
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16
        ).to(device)
        dtype_str = "bfloat16"
        print("Using CPU with bfloat16")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    num_model_layers = model.config.num_hidden_layers
    num_sae_layers = min(num_model_layers, MAX_SAE_LAYER + 1)
    print(f"Model: {MODEL_ID} ({num_model_layers} layers, SAEs for 0-{MAX_SAE_LAYER})")
    print(f"Experiment scope: layers {START_LAYER}-{END_LAYER - 1} ({END_LAYER - START_LAYER} layers)")

    prompts = YES_PROMPTS + NO_PROMPTS
    print(f"\nContrastive prompts: {len(YES_PROMPTS)} Yes + {len(NO_PROMPTS)} No")
    print(f"  Yes: {YES_PROMPTS[0]}")
    print(f"  No:  {NO_PROMPTS[0]}")

    # Token IDs for logit attribution
    yes_token_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode(" No", add_special_tokens=False)[0]
    logit_token_ids = [yes_token_id, no_token_id]
    logit_token_labels = [" Yes", " No"]
    print(f"Logit tokens: {list(zip(logit_token_labels, logit_token_ids))}")

    # Use first Yes prompt for gradient computation
    inputs = tokenizer.encode(YES_PROMPTS[0], return_tensors="pt", add_special_tokens=True).to(device)

    # Get contrastive features across ALL SAE layers (needed for gradient computation)
    print("\nFinding contrastive features (Yes vs No)...")
    activated_features, activation_frequencies = get_contrastive_features(
        YES_PROMPTS, NO_PROMPTS, tokenizer, model, device, n_features=N_FEATURES,
    )
    print(f"Features per layer (layers {START_LAYER}-{END_LAYER-1}): "
          f"{[len(activated_features[i]) for i in range(START_LAYER, END_LAYER)]}")

    # Only compute gradients for our scoped layers (adjacent pairs)
    layer_pairs = [(i, j) for i in range(START_LAYER, END_LAYER)
                   for j in range(i + 1, END_LAYER)]
    n_pairs = len(layer_pairs)
    print(f"\nComputing gradient matrices for {n_pairs} layer pairs "
          f"(layers {START_LAYER}-{END_LAYER - 1})...")

    gradient_matrices = []
    logit_gradients: dict[int, torch.Tensor] = {}
    total_start = time.time()
    pairs_done = 0

    for up_layer in range(START_LAYER, END_LAYER - 1):
        up_feats = activated_features[up_layer][:N_FEATURES]
        n_up = len(up_feats)

        downstream_pairs = []
        for down_layer in range(up_layer + 1, END_LAYER):
            down_feats = activated_features[down_layer][:N_FEATURES]
            downstream_pairs.append((down_layer, down_feats))

        n_down_layers = len(downstream_pairs)
        print(f"\n[Upstream L{up_layer}] {n_up} features -> {n_down_layers} downstream layers "
              f"({pairs_done + 1}..{pairs_done + n_down_layers}/{n_pairs})")

        t0 = time.time()
        batch_results, logit_grad_matrix = compute_gradient_matrices_batch(
            inputs, up_layer, downstream_pairs, up_feats, model, verbose=True,
            logit_token_ids=logit_token_ids,
        )
        elapsed = time.time() - t0

        for grad_matrix in batch_results:
            gradient_matrices.append(grad_matrix.cpu())
        if logit_grad_matrix is not None:
            logit_gradients[up_layer] = logit_grad_matrix.cpu()
        pairs_done += n_down_layers

        print(f"  {elapsed:.1f}s for {n_down_layers} pairs "
              f"({elapsed / n_down_layers:.2f}s/pair avg)")

    total_elapsed = time.time() - total_start
    print(f"\n--- Total gradient computation: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min) "
          f"for {n_pairs} pairs ---")

    # Quick sanity check — print gradient stats
    print("\n=== Gradient Statistics ===")
    for (up, down), mat in zip(layer_pairs, gradient_matrices):
        abs_mat = mat.abs()
        print(f"  L{up}->L{down}: shape={mat.shape}, "
              f"max={abs_mat.max():.4f}, mean={abs_mat.mean():.6f}, "
              f"nonzero={mat.count_nonzero()}/{mat.numel()}")

    if logit_gradients:
        print("\n=== Logit Gradient Statistics ===")
        for layer_idx, grad_mat in logit_gradients.items():
            for t_idx, label in enumerate(logit_token_labels):
                row = grad_mat[t_idx]
                print(f"  L{layer_idx} -> '{label}': max={row.abs().max():.4f}, "
                      f"mean={row.abs().mean():.6f}, nonzero={row.count_nonzero()}/{row.numel()}")

    # Free model memory before export
    del model, tokenizer, inputs
    _sae_cache.clear()
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"results/circuit_quick_{timestamp}.json"

    metadata = {
        "model": MODEL_ID,
        "prompts": prompts,
        "feature_selection": "contrastive (|mean_act_yes - mean_act_no|)",
        "yes_prompts": YES_PROMPTS,
        "no_prompts": NO_PROMPTS,
        "n_features_per_layer": N_FEATURES,
        "layer_range": f"{START_LAYER}-{END_LAYER - 1}",
        "device": str(device),
        "dtype": dtype_str,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\nExporting JSON to {json_path}...")
    json_out = export_circuit_json(
        metadata, activated_features, gradient_matrices, layer_pairs, json_path,
        activation_frequencies=activation_frequencies,
        logit_gradients=logit_gradients,
        logit_token_labels=logit_token_labels,
        logit_token_ids=logit_token_ids,
    )

    print(f"Generating HTML visualizer...")
    html_out = export_circuit_html(json_out)

    print(f"\nDone!")
    print(f"  JSON: {json_out}")
    print(f"  HTML: {html_out}")
    print(f"\nOpen the HTML file in your browser to explore the circuit.")
