"""Run addition circuit experiment on Gemma 2 2B.

Contrastive setup: "XX+YY=" (addition) vs "XX-YY=" (subtraction).
Generates 50 prompt pairs programmatically with a fixed seed.

Usage:
    PYTHONPATH=. .venv/bin/python -m experiments.run_addition_gemma2_2b
"""

import gc
import random
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Switch to Gemma 2 2B BEFORE importing anything that reads data.py at import time
from feature_circuit_discovery.data import set_model
set_model("gemma-2-2b")

from experiments.feature_grad_exp_optimized import (
    compute_gradient_matrices_batch,
    compute_token_gradients,
    get_contrastive_features,
    load_sae,
    _sae_cache,
)
from feature_circuit_discovery.data import MODEL_ID, MAX_SAE_LAYER, canonical_sae_filenames
from feature_circuit_discovery.export import export_circuit_json, export_circuit_html

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_FEATURES = 50       # features per layer
N_PROMPTS = 50        # addition/subtraction prompt pairs
SEED = 42


def generate_addition_prompts(n: int, seed: int) -> tuple[list[str], list[str]]:
    """Generate n addition and subtraction prompt pairs.

    Picks random 2-digit numbers with a > b and a + b < 100, so both
    the sum and difference are 2-digit (important for single-token logits).

    Returns:
        Tuple of (addition_prompts, subtraction_prompts).
    """
    rng = random.Random(seed)
    add_prompts = []
    sub_prompts = []
    for _ in range(n):
        # a in [11, 49], b in [10, min(a-1, 99-a)] ensures:
        #   a + b < 100 (2-digit sum)
        #   a > b >= 10 (non-negative 2-digit difference)
        a = rng.randint(20, 49)
        b = rng.randint(10, min(a - 1, 99 - a))
        add_prompts.append(f"{a}+{b}=")
        sub_prompts.append(f"{a}-{b}=")
    return add_prompts, sub_prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Model: {MODEL_ID}")
    print(f"SAE layers: 0-{MAX_SAE_LAYER} ({MAX_SAE_LAYER + 1} layers)")
    print(f"SAE files: {canonical_sae_filenames[0]} ... {canonical_sae_filenames[-1]}")

    # Generate prompts
    prompts_add, prompts_sub = generate_addition_prompts(N_PROMPTS, SEED)

    print(f"\nAddition prompts: {N_PROMPTS} pairs")
    print(f"  Example add: {prompts_add[0]}")
    print(f"  Example sub: {prompts_sub[0]}")

    # Device selection: MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16
        ).to(device)
        dtype_str = "bfloat16"
        print("\nUsing MPS (Apple Silicon GPU) with bfloat16")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="auto"
        )
        dtype_str = "bfloat16"
        print("\nUsing CUDA with bfloat16")
    else:
        device = torch.device("cpu")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16
        ).to(device)
        dtype_str = "bfloat16"
        print("\nUsing CPU with bfloat16")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    num_layers = min(model.config.num_hidden_layers, MAX_SAE_LAYER + 1)
    print(f"Model layers: {model.config.num_hidden_layers}, using {num_layers} (SAE coverage)")

    # Use first addition prompt for gradient computation
    inputs = tokenizer.encode(
        prompts_add[0], return_tensors="pt", add_special_tokens=True
    ).to(device)

    # Get contrastive features — addition vs subtraction
    print("\nFinding contrastive features (addition vs subtraction)...")
    activated_features, activation_frequencies = get_contrastive_features(
        prompts_add, prompts_sub, tokenizer, model, device, n_features=N_FEATURES,
    )
    print(f"Features per layer (first 5): {[len(f) for f in activated_features[:5]]}")

    # Token attribution: which input tokens drive each SAE feature?
    print("\nComputing token attribution gradients...")
    token_gradients = compute_token_gradients(
        inputs, activated_features, model, n_features=N_FEATURES, verbose=True,
    )
    input_token_ids = inputs[0].cpu().tolist()
    input_token_labels = [tokenizer.decode([tid]) for tid in input_token_ids]
    print(f"Token labels: {input_token_labels}")

    # Logit attribution: track all 10 digit tokens (0-9) at TWO positions:
    #   Position 1 (d1): after "=" — which tens-digit does the model predict?
    #   Position 2 (d2): after "=X" — which units-digit does the model predict?
    first_add = prompts_add[0]  # e.g. "34+18="
    parts = first_add.rstrip("=").split("+")
    a_val, b_val = int(parts[0]), int(parts[1])
    sum_val = a_val + b_val
    sum_first_digit = str(sum_val)[0]
    sum_second_digit = str(sum_val)[1] if len(str(sum_val)) > 1 else "0"

    # Build digit token IDs for 0-9
    digit_token_ids = []
    for d in range(10):
        ids = tokenizer.encode(str(d), add_special_tokens=False)
        digit_token_ids.append(ids[-1])  # last token = the digit itself

    # First position labels: d1:0 through d1:9
    logit_token_ids = digit_token_ids
    logit_token_labels = [f"d1:{d}" for d in range(10)]

    print(f"Sum: {sum_val} (d1={sum_first_digit}, d2={sum_second_digit})")
    print(f"Digit token IDs: {digit_token_ids}")
    print(f"First position logit labels: {logit_token_labels}")

    # All upstream → downstream layer pairs
    layer_pairs = [(i, j) for i in range(num_layers) for j in range(i + 1, num_layers)]
    n_pairs = len(layer_pairs)
    print(f"\nComputing gradient matrices for {n_pairs} layer pairs "
          f"({num_layers} layers, all upstream->downstream)...")
    print(f"Using batched computation: 1 forward pass per upstream layer "
          f"({num_layers - 1} forward passes instead of {n_pairs})")

    gradient_matrices = []
    logit_gradients: dict[int, torch.Tensor] = {}
    total_start = time.time()
    pairs_done = 0

    # Group by upstream layer for shared forward passes
    for up_layer in range(num_layers - 1):
        up_feats = activated_features[up_layer][:N_FEATURES]
        n_up = len(up_feats)

        downstream_pairs = []
        for down_layer in range(up_layer + 1, num_layers):
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
    print(f"\n--- First-digit gradient computation: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min) "
          f"for {n_pairs} pairs ---")

    # --- Second-digit pass (d2): "XX+YY=Z" where Z is the correct first digit ---
    print(f"\n{'='*60}")
    print(f"Second-digit pass: appending '{sum_first_digit}' to prompt -> \"{prompts_add[0]}{sum_first_digit}\"")
    print(f"{'='*60}")

    inputs_d2 = tokenizer.encode(
        prompts_add[0] + sum_first_digit, return_tensors="pt", add_special_tokens=True
    ).to(device)

    logit_gradients_d2: dict[int, torch.Tensor] = {}
    d2_start = time.time()

    for up_layer in range(num_layers - 1):
        up_feats = activated_features[up_layer][:N_FEATURES]
        n_up = len(up_feats)

        print(f"\n[D2 Upstream L{up_layer}] {n_up} features -> logit only (no downstream pairs)")

        t0 = time.time()
        _, logit_grad_matrix_d2 = compute_gradient_matrices_batch(
            inputs_d2, up_layer, downstream_pairs=[], upstream_features=up_feats,
            model=model, verbose=True, logit_token_ids=digit_token_ids,
        )
        elapsed = time.time() - t0

        if logit_grad_matrix_d2 is not None:
            logit_gradients_d2[up_layer] = logit_grad_matrix_d2.cpu()

        print(f"  {elapsed:.1f}s")

    d2_elapsed = time.time() - d2_start
    print(f"\n--- Second-digit gradient computation: {d2_elapsed:.0f}s ({d2_elapsed/60:.1f} min) ---")

    # --- Merge d1 and d2 logit gradients ---
    merged_logit_gradients: dict[int, torch.Tensor] = {}
    for layer in sorted(set(logit_gradients.keys()) | set(logit_gradients_d2.keys())):
        d1 = logit_gradients.get(layer)
        d2 = logit_gradients_d2.get(layer)
        if d1 is not None and d2 is not None:
            merged_logit_gradients[layer] = torch.cat([d1, d2], dim=0)  # (20, n_feats)
        elif d1 is not None:
            # Pad with zeros for d2
            merged_logit_gradients[layer] = torch.cat(
                [d1, torch.zeros_like(d1)], dim=0
            )
        elif d2 is not None:
            # Pad with zeros for d1
            merged_logit_gradients[layer] = torch.cat(
                [torch.zeros_like(d2), d2], dim=0
            )

    merged_logit_labels = [f"d1:{d}" for d in range(10)] + [f"d2:{d}" for d in range(10)]
    merged_logit_token_ids = digit_token_ids + digit_token_ids

    total_elapsed = time.time() - total_start
    print(f"\n--- Total computation: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min) ---")

    # Free model memory before export
    del model, tokenizer, inputs, inputs_d2
    _sae_cache.clear()
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"results/addition_gemma2_2b_{timestamp}.json"

    all_prompts = prompts_add + prompts_sub
    metadata = {
        "model": MODEL_ID,
        "sae_repo": "google/gemma-scope-2b-pt-res",
        "prompts": all_prompts,
        "n_features_per_layer": N_FEATURES,
        "feature_selection": "contrastive (|mean_act_addition - mean_act_subtraction|)",
        "addition_prompts": prompts_add,
        "subtraction_prompts": prompts_sub,
        "device": str(device),
        "dtype": dtype_str,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\nExporting JSON to {json_path}...")
    json_out = export_circuit_json(
        metadata, activated_features, gradient_matrices, layer_pairs, json_path,
        activation_frequencies=activation_frequencies,
        logit_gradients=merged_logit_gradients,
        logit_token_labels=merged_logit_labels,
        logit_token_ids=merged_logit_token_ids,
        token_gradients=token_gradients,
        token_labels=input_token_labels,
        token_ids=input_token_ids,
    )

    print(f"Generating HTML visualizer...")
    html_out = export_circuit_html(json_out)

    print(f"\nDone!")
    print(f"  JSON: {json_out}")
    print(f"  HTML: {html_out}")
    print(f"\nOpen the HTML file in your browser to explore the circuit.")
