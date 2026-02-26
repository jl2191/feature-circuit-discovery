"""Run number comparison circuit experiment on Gemma 2 2B.

Contrastive setup: "Is X>Y? " where X,Y are 3-digit numbers.
Group A (Yes): 12 prompts where X > Y
Group B (No): 13 prompts where X <= Y (including 1 equality)

Balanced across 4 buckets:
  A: Hundreds digit decides (4 pairs = 8 prompts)
  B: Tens digit decides, hundreds tie (4 pairs = 8 prompts)
  C: Ones digit decides, hundreds+tens tie (4 pairs = 8 prompts)
  D: Equality (1 prompt, No)

Usage:
    PYTHONPATH=. .venv/bin/python -m experiments.run_comparison_gemma2_2b
"""

import gc
import random
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
SEED = 42


def generate_comparison_prompts(seed: int = 42) -> tuple[list[str], list[str]]:
    """Generate balanced 3-digit number comparison prompts.

    Returns 12 Yes prompts and 13 No prompts from 4 buckets:
      A: Hundreds digit decides (4 pairs -> 4 Yes + 4 No)
      B: Tens digit decides (4 pairs -> 4 Yes + 4 No)
      C: Ones digit decides (4 pairs -> 4 Yes + 4 No)
      D: Equality (1 No)

    Each pair (a, b) with a > b generates both "Is a>b? " (Yes) and
    "Is b>a? " (No), ensuring digit balance across groups.
    """
    rng = random.Random(seed)
    yes_prompts = []
    no_prompts = []

    # Bucket A: hundreds differ (4 pairs)
    hundreds_pairs = [(5, 3), (7, 4), (8, 2), (6, 1)]
    for h_big, h_small in hundreds_pairs:
        t, o = rng.randint(1, 9), rng.randint(0, 9)
        a = h_big * 100 + t * 10 + o
        b = h_small * 100 + t * 10 + o  # same tens/ones
        yes_prompts.append(f"Is {a}>{b}? ")
        no_prompts.append(f"Is {b}>{a}? ")

    # Bucket B: hundreds same, tens differ (4 pairs)
    hundreds_b = [3, 5, 7, 9]
    tens_pairs = [(6, 2), (8, 4), (7, 3), (5, 1)]
    for h, (t_big, t_small) in zip(hundreds_b, tens_pairs):
        o = rng.randint(0, 9)
        a = h * 100 + t_big * 10 + o
        b = h * 100 + t_small * 10 + o  # same ones
        yes_prompts.append(f"Is {a}>{b}? ")
        no_prompts.append(f"Is {b}>{a}? ")

    # Bucket C: hundreds+tens same, ones differ (4 pairs)
    ht_pairs = [(4, 5), (6, 3), (8, 7), (2, 9)]  # (hundreds, tens)
    ones_pairs = [(7, 2), (9, 4), (6, 1), (8, 3)]
    for (h, t), (o_big, o_small) in zip(ht_pairs, ones_pairs):
        a = h * 100 + t * 10 + o_big
        b = h * 100 + t * 10 + o_small
        yes_prompts.append(f"Is {a}>{b}? ")
        no_prompts.append(f"Is {b}>{a}? ")

    # Bucket D: equality (1 prompt -> No)
    eq = rng.randint(100, 999)
    no_prompts.append(f"Is {eq}>{eq}? ")

    return yes_prompts, no_prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Model: {MODEL_ID}")
    print(f"SAE layers: 0-{MAX_SAE_LAYER} ({MAX_SAE_LAYER + 1} layers)")
    print(f"SAE files: {canonical_sae_filenames[0]} ... {canonical_sae_filenames[-1]}")

    # Generate prompts
    prompts_yes, prompts_no = generate_comparison_prompts(SEED)

    print(f"\nComparison prompts: {len(prompts_yes)} Yes + {len(prompts_no)} No = {len(prompts_yes) + len(prompts_no)} total")
    print(f"  Bucket A (hundreds decide):  {prompts_yes[:4]}")
    print(f"  Bucket B (tens decide):      {prompts_yes[4:8]}")
    print(f"  Bucket C (ones decide):      {prompts_yes[8:12]}")
    print(f"  Bucket D (equality):         {prompts_no[-1]}")

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

    # Representative prompt: use a Bucket C prompt (ones-digit decides)
    # This forces the model to process all three digit positions
    representative_prompt = prompts_yes[8]  # first Bucket C Yes prompt
    print(f"\nRepresentative prompt: {repr(representative_prompt)}")

    inputs = tokenizer.encode(
        representative_prompt, return_tensors="pt", add_special_tokens=True
    ).to(device)

    # Get contrastive features — Yes vs No
    print("\nFinding contrastive features (Yes vs No)...")
    activated_features, activation_frequencies = get_contrastive_features(
        prompts_yes, prompts_no, tokenizer, model, device, n_features=N_FEATURES,
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

    # Logit attribution: track Yes and No tokens
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    logit_token_ids = [yes_token_id, no_token_id]
    logit_token_labels = ["Yes", "No"]
    print(f"Logit tokens: Yes={yes_token_id}, No={no_token_id}")

    # All upstream -> downstream layer pairs
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
    json_path = f"results/comparison_gemma2_2b_{timestamp}.json"

    all_prompts = prompts_yes + prompts_no
    metadata = {
        "model": MODEL_ID,
        "sae_repo": "google/gemma-scope-2b-pt-res",
        "prompts": all_prompts,
        "n_features_per_layer": N_FEATURES,
        "feature_selection": "contrastive (|mean_act_yes - mean_act_no|)",
        "yes_prompts": prompts_yes,
        "no_prompts": prompts_no,
        "representative_prompt": representative_prompt,
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
