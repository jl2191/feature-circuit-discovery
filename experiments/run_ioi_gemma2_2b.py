"""Run IOI (Indirect Object Identification) feature circuit experiment on Gemma 2 2B.

Usage:
    poetry run python -m experiments.run_ioi_gemma2_2b
"""

import gc
import json
import re
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
N_PROMPTS = 25        # IOI prompts to use
IOI_DATA_PATH = "datasets/ioi/ioi_test_100.json"


def load_ioi_data(path: str, n: int) -> tuple[list[str], list[str], list[str]]:
    """Load IOI prompts, sentences, and solutions from JSON.

    Returns:
        Tuple of (prompts, sentences, solutions) — each a list of n strings.
    """
    with open(path) as f:
        data = json.load(f)
    return data["prompts"][:n], data["sentences"][:n], data["solutions"][:n]


def swap_names_in_prompt(prompt: str, sentence: str) -> str:
    """Create a contrastive prompt by swapping the two names in the IOI prompt.

    The IOI task has exactly two names. Swapping them flips which name is the
    indirect object vs. the subject, creating a natural contrastive pair.
    """
    # Extract the two names from the full sentence (which has both names clearly)
    # IOI sentences follow patterns like "X and Y ...", "Friends X and Y ..."
    # Find all capitalized words that appear as names (not sentence-initial common words)
    # Strategy: find the two names by looking at the solution and the other name
    # The prompt ends with "... gave [something] to" — the solution is the last name.
    # We need to find both names in the sentence.

    # Find all proper name candidates: capitalized words not at sentence start
    # that aren't common words
    common_words = {
        "Friends", "The", "After", "While", "When", "Afterwards",
        "Answer", "Is", "A", "I", "It",
    }

    # Extract names by finding the pattern "Name1 and Name2" in the sentence
    match = re.search(r"(\b[A-Z][a-z]+\b)\s+and\s+(\b[A-Z][a-z]+\b)", sentence)
    if not match:
        # Fallback: return prompt unchanged
        return prompt

    name1, name2 = match.group(1), match.group(2)

    # Swap name1 and name2 in the prompt
    # Use a placeholder to avoid double-swapping
    placeholder = "___PLACEHOLDER___"
    swapped = prompt.replace(name1, placeholder)
    swapped = swapped.replace(name2, name1)
    swapped = swapped.replace(placeholder, name2)

    return swapped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Model: {MODEL_ID}")
    print(f"SAE layers: 0-{MAX_SAE_LAYER} ({MAX_SAE_LAYER + 1} layers)")
    print(f"SAE files: {canonical_sae_filenames[0]} ... {canonical_sae_filenames[-1]}")

    # Load IOI data
    prompts_ioi, sentences_ioi, solutions_ioi = load_ioi_data(IOI_DATA_PATH, N_PROMPTS)

    # Build contrastive pairs: swap names to flip IO vs subject
    prompts_swapped = [
        swap_names_in_prompt(p, s) for p, s in zip(prompts_ioi, sentences_ioi)
    ]

    print(f"\nIOI prompts: {N_PROMPTS} original + {N_PROMPTS} name-swapped")
    print(f"  Original:  {prompts_ioi[0]} -> {solutions_ioi[0]}")
    print(f"  Swapped:   {prompts_swapped[0]}")

    # Device selection: MPS > CUDA > CPU
    # Use bfloat16 everywhere — Gemma 2 2B is ~10GB in float32, ~5GB in bfloat16
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

    # Batch all prompts for gradient computation (label-aligned averaging)
    all_prompts = prompts_ioi + prompts_swapped
    tokenized = tokenizer(
        all_prompts, return_tensors="pt", add_special_tokens=True, padding=True
    )
    inputs = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    # Label-aligned signs: +1 for IOI (original), -1 for name-swapped
    prompt_signs = torch.cat([
        torch.ones(len(prompts_ioi), device=device),
        -torch.ones(len(prompts_swapped), device=device),
    ])
    print(f"\nGradient input: {len(all_prompts)} prompts, shape {tuple(inputs.shape)}")
    print(f"Label-aligned: {len(prompts_ioi)} IOI (+1), {len(prompts_swapped)} swapped (-1)")

    # Get contrastive features — IOI vs name-swapped
    print("\nFinding contrastive features (IOI vs name-swapped)...")
    activated_features, activation_frequencies = get_contrastive_features(
        prompts_ioi, prompts_swapped, tokenizer, model, device, n_features=N_FEATURES,
    )
    print(f"Features per layer (first 5): {[len(f) for f in activated_features[:5]]}")

    # Logit attribution: per-prompt IO and S name tokens
    # For original prompts: IO = solution name, S = the other name (giver)
    # For swapped prompts: roles flip — IO becomes the original S, S becomes the original IO
    n_total = len(all_prompts)
    io_token_ids_list = []  # per-prompt IO name token IDs
    subj_token_ids_list = []  # per-prompt S name token IDs

    for idx in range(N_PROMPTS):
        io_name = solutions_ioi[idx]
        match = re.search(r"(\b[A-Z][a-z]+\b)\s+and\s+(\b[A-Z][a-z]+\b)", sentences_ioi[idx])
        if match:
            name1, name2 = match.group(1), match.group(2)
            subj_name = name1 if name1 != io_name else name2
        else:
            subj_name = io_name  # fallback

        io_tid = tokenizer.encode(f" {io_name}", add_special_tokens=False)[0]
        subj_tid = tokenizer.encode(f" {subj_name}", add_special_tokens=False)[0]

        # Original prompt: IO = solution, S = giver
        io_token_ids_list.append(io_tid)
        subj_token_ids_list.append(subj_tid)

    for idx in range(N_PROMPTS):
        # Swapped prompt: names are swapped, so IO and S token IDs also swap
        io_token_ids_list.append(subj_token_ids_list[idx])
        subj_token_ids_list.append(io_token_ids_list[idx])

    # Shape: (2, n_total) — slot 0 = IO name, slot 1 = S name
    logit_token_ids_per_prompt = torch.tensor(
        [io_token_ids_list, subj_token_ids_list], dtype=torch.long, device=device,
    )
    logit_token_labels = ["IO name", "S name"]
    # Representative token IDs for export tooltip (from first prompt)
    first_io_token_id = io_token_ids_list[0]
    first_subj_token_id = subj_token_ids_list[0]
    logit_token_ids = [first_io_token_id, first_subj_token_id]
    print(f"Logit tokens (per-prompt): {logit_token_ids_per_prompt.shape}")
    print(f"  Representative: {list(zip(logit_token_labels, logit_token_ids))}")

    # All upstream → downstream layer pairs
    layer_pairs = [(i, j) for i in range(num_layers) for j in range(i + 1, num_layers)]
    n_pairs = len(layer_pairs)
    print(f"\nComputing gradient matrices for {n_pairs} layer pairs "
          f"({num_layers} layers, all upstream→downstream)...")
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
            logit_token_ids_per_prompt=logit_token_ids_per_prompt,
            prompt_signs=prompt_signs, attention_mask=attention_mask,
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
    del model, tokenizer, inputs, attention_mask
    _sae_cache.clear()
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"results/ioi_gemma2_2b_{timestamp}.json"

    metadata = {
        "model": MODEL_ID,
        "sae_repo": "google/gemma-scope-2b-pt-res",
        "prompts": all_prompts,
        "n_features_per_layer": N_FEATURES,
        "feature_selection": "contrastive (|mean_act_ioi - mean_act_swapped|)",
        "gradient_method": f"label-aligned average over {len(all_prompts)} prompts (IOI=+1, swapped=-1)",
        "ioi_prompts": prompts_ioi,
        "swapped_prompts": prompts_swapped,
        "solutions": solutions_ioi,
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
