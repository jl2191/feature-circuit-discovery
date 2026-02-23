"""Run feature circuit experiment and export results as JSON + interactive HTML.

Usage:
    poetry run python -m experiments.run_and_export
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
# Configuration
# ---------------------------------------------------------------------------
N_FEATURES = 50  # features per layer (capped by availability)

# Contrastive prompt pairs — mirrored digits to control for digit-level features
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
    num_layers = min(model.config.num_hidden_layers, MAX_SAE_LAYER + 1)

    prompts = YES_PROMPTS + NO_PROMPTS
    print(f"Contrastive prompts: {len(YES_PROMPTS)} Yes + {len(NO_PROMPTS)} No")
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

    # Get contrastive features — ranked by |mean_act_yes - mean_act_no| at last token
    print("\nFinding contrastive features (Yes vs No)...")
    activated_features, activation_frequencies = get_contrastive_features(
        YES_PROMPTS, NO_PROMPTS, tokenizer, model, device, n_features=N_FEATURES,
    )
    print(f"Features per layer (first 5): {[len(f) for f in activated_features[:5]]}")

    # All upstream → downstream layer pairs (not just adjacent)
    layer_pairs = [(i, j) for i in range(num_layers) for j in range(i + 1, num_layers)]
    n_pairs = len(layer_pairs)
    print(f"\nComputing gradient matrices for {n_pairs} layer pairs "
          f"({num_layers} layers, all upstream→downstream)...")
    print(f"Using batched computation: 1 forward pass per upstream layer "
          f"({num_layers - 1} forward passes instead of {n_pairs})")

    gradient_matrices = []
    logit_gradients: dict[int, torch.Tensor] = {}  # upstream_layer -> (2, n_feats)
    total_start = time.time()
    pairs_done = 0

    # Group by upstream layer for shared forward passes
    for up_layer in range(num_layers - 1):
        up_feats = activated_features[up_layer][:N_FEATURES]
        n_up = len(up_feats)

        # Build list of (downstream_layer, downstream_features) for this upstream
        downstream_pairs = []
        for down_layer in range(up_layer + 1, num_layers):
            down_feats = activated_features[down_layer][:N_FEATURES]
            downstream_pairs.append((down_layer, down_feats))

        n_down_layers = len(downstream_pairs)
        print(f"\n[Upstream L{up_layer}] {n_up} features → {n_down_layers} downstream layers "
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

    # Free model memory before export
    del model, tokenizer, inputs
    _sae_cache.clear()
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"results/circuit_{timestamp}.json"

    metadata = {
        "model": MODEL_ID,
        "prompts": prompts,
        "feature_selection": "contrastive (|mean_act_yes - mean_act_no|)",
        "yes_prompts": YES_PROMPTS,
        "no_prompts": NO_PROMPTS,
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
