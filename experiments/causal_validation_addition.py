"""Causal validation of the addition circuit.

Ablates discovered circuit features and measures impact on addition accuracy.
Compares:
  - All-position ablation vs last-token-only ablation
  - Three ranking methods: total |grad|, outgoing |grad|, causal (out−in)
  - Random feature ablation as control

Usage:
    PYTHONPATH=. .venv/bin/python experiments/causal_validation_addition.py
"""

import gc
import json
import random
import time
from collections import Counter, defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feature_circuit_discovery.data import set_model
set_model("gemma-2-2b")

from feature_circuit_discovery.core import load_sae, _sae_cache
from feature_circuit_discovery.data import MODEL_ID, canonical_sae_filenames


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CIRCUIT_RESULTS_PATH = "results/addition_gemma2_2b_20260225_234921.json"
N_EVAL_PROMPTS = 100
SEED = 123  # different seed from training prompts (42) to test generalization
TOP_K_FEATURES = 10  # how many features to ablate per layer
N_RANDOM_TRIALS = 5


def generate_eval_prompts(n: int, seed: int) -> list[tuple[str, int]]:
    """Generate addition prompts with known answers, disjoint from training set."""
    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        a = rng.randint(10, 49)
        b = rng.randint(10, min(a, 99 - a))
        prompts.append((f"{a}+{b}=", a + b))
    return prompts


def get_model_answer(model, tokenizer, prompt: str, device) -> tuple[int | None, str]:
    """Run the model on a prompt and extract the predicted number."""
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        output = model.generate(
            inputs, max_new_tokens=3, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    digits = ""
    for ch in generated:
        if ch.isdigit():
            digits += ch
        else:
            break
    try:
        return int(digits), generated
    except ValueError:
        return None, generated


def eval_accuracy(model, tokenizer, prompts, device) -> tuple[float, list[dict]]:
    """Evaluate addition accuracy. Returns (accuracy, per-prompt results)."""
    results = []
    correct = 0
    for prompt, answer in prompts:
        pred, raw = get_model_answer(model, tokenizer, prompt, device)
        is_correct = pred == answer
        if is_correct:
            correct += 1
        results.append({
            "prompt": prompt, "answer": answer,
            "predicted": pred, "raw": raw, "correct": is_correct,
        })
    return correct / len(prompts), results


# ---------------------------------------------------------------------------
# Feature ranking methods
# ---------------------------------------------------------------------------

def load_circuit_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_feature_scores(data: dict) -> dict[tuple[int, int], dict[str, float]]:
    """Compute total, outgoing, incoming, and causal scores for every feature."""
    outgoing: Counter[tuple[int, int]] = Counter()
    incoming: Counter[tuple[int, int]] = Counter()

    for pair in data["layer_pairs"]:
        mat = pair["gradient_matrix"]
        up_feats = pair["upstream_feature_ids"]
        down_feats = pair["downstream_feature_ids"]
        for j, uid in enumerate(up_feats):
            s = sum(abs(mat[i][j]) for i in range(len(down_feats)))
            outgoing[(pair["upstream_layer"], uid)] += s
        for i, did in enumerate(down_feats):
            s = sum(abs(mat[i][j]) for j in range(len(up_feats)))
            incoming[(pair["downstream_layer"], did)] += s

    all_keys = set(outgoing.keys()) | set(incoming.keys())
    scores = {}
    for key in all_keys:
        out = outgoing[key]
        inc = incoming[key]
        scores[key] = {
            "total": out + inc,
            "outgoing": out,
            "incoming": inc,
            "causal": max(0, out - inc),
        }
    return scores


def select_top_features(
    scores: dict[tuple[int, int], dict[str, float]],
    method: str,
    top_k: int,
) -> dict[int, list[int]]:
    """Select top_k features per layer by the given ranking method."""
    layer_feats: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (layer, fid), s in scores.items():
        layer_feats[layer].append((fid, s[method]))

    result = {}
    for layer, feats in layer_feats.items():
        feats.sort(key=lambda x: x[1], reverse=True)
        top_ids = [fid for fid, _ in feats[:top_k]]
        if top_ids:
            result[layer] = top_ids
    return result


# ---------------------------------------------------------------------------
# Ablation hooks
# ---------------------------------------------------------------------------

def make_ablation_hooks(
    circuit_features: dict[int, list[int]],
    saes: dict[int, object],
    device,
    last_token_only: bool = False,
):
    """Create forward hooks that zero out specified SAE features.

    Args:
        circuit_features: layer_idx -> list of feature IDs to ablate.
        saes: layer_idx -> loaded SAE.
        device: torch device.
        last_token_only: If True, only ablate at the last sequence position.
            If False, ablate at all positions.
    """
    hooks = []

    for layer_idx, feature_ids in circuit_features.items():
        sae = saes[layer_idx]
        feat_tensor = torch.tensor(feature_ids, device=device)

        def make_hook(sae, feat_idx, last_only):
            def hook_fn(module, input, output):
                residual = output[0] if isinstance(output, tuple) else output

                if last_only:
                    # Only ablate the last token position
                    last_pos = residual.shape[1] - 1
                    last_residual = residual[:, last_pos:last_pos+1, :]

                    sae_acts = sae.encode(last_residual.float())
                    original_recon = sae.decode(sae_acts)

                    sae_acts_ablated = sae_acts.clone()
                    sae_acts_ablated[:, :, feat_idx] = 0.0
                    ablated_recon = sae.decode(sae_acts_ablated)

                    delta = (ablated_recon - original_recon).to(residual.dtype)
                    modified = residual.clone()
                    modified[:, last_pos:last_pos+1, :] += delta
                else:
                    # Ablate at all positions
                    sae_acts = sae.encode(residual.float())
                    original_recon = sae.decode(sae_acts)

                    sae_acts_ablated = sae_acts.clone()
                    sae_acts_ablated[:, :, feat_idx] = 0.0
                    ablated_recon = sae.decode(sae_acts_ablated)

                    delta = (ablated_recon - original_recon).to(residual.dtype)
                    modified = residual + delta

                return (modified,) + output[1:] if isinstance(output, tuple) else modified
            return hook_fn

        hooks.append((layer_idx, make_hook(sae, feat_tensor, last_token_only)))

    return hooks


def run_with_ablation(model, hooks):
    """Register hooks, return handles."""
    handles = []
    for layer_idx, hook_fn in hooks:
        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
        handles.append(handle)
    return handles


def run_ablation_experiment(
    label: str,
    circuit_features: dict[int, list[int]],
    saes: dict[int, object],
    model, tokenizer, eval_prompts, device,
    last_token_only: bool,
) -> tuple[float, list[dict]]:
    """Run one ablation experiment and print results. Returns (accuracy, results)."""
    hooks = make_ablation_hooks(circuit_features, saes, device, last_token_only=last_token_only)
    handles = run_with_ablation(model, hooks)
    acc, results = eval_accuracy(model, tokenizer, eval_prompts, device)
    for h in handles:
        h.remove()
    return acc, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("CAUSAL VALIDATION: Addition Circuit")
    print("=" * 70)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\nLoading model ({MODEL_ID}) to {device}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Eval prompts
    eval_prompts = generate_eval_prompts(N_EVAL_PROMPTS, SEED)
    print(f"Eval prompts: {N_EVAL_PROMPTS} (seed={SEED}, disjoint from training seed=42)")
    print(f"  Examples: {eval_prompts[0]}, {eval_prompts[1]}, {eval_prompts[2]}")

    # Load circuit data and compute scores
    data = load_circuit_data(CIRCUIT_RESULTS_PATH)
    scores = compute_feature_scores(data)

    # Pre-load all SAEs
    all_layers = set()
    for (layer, _) in scores:
        all_layers.add(layer)
    print(f"\nLoading SAEs for {len(all_layers)} layers...")
    saes = {}
    for layer_idx in sorted(all_layers):
        saes[layer_idx] = load_sae(canonical_sae_filenames[layer_idx], device)

    # ===================================================================
    # BASELINE
    # ===================================================================
    print("\n" + "=" * 70)
    print("BASELINE (no ablation)")
    print("=" * 70)
    t0 = time.time()
    baseline_acc, baseline_results = eval_accuracy(model, tokenizer, eval_prompts, device)
    print(f"Accuracy: {baseline_acc:.1%} ({int(baseline_acc * N_EVAL_PROMPTS)}/{N_EVAL_PROMPTS})")
    print(f"Time: {time.time() - t0:.1f}s")
    errors = [r for r in baseline_results if not r["correct"]]
    if errors:
        print(f"Baseline errors ({len(errors)}):")
        for r in errors[:5]:
            print(f"  {r['prompt']} expected={r['answer']}, got={r['predicted']}")

    # ===================================================================
    # COMPARE RANKING METHODS × ABLATION MODES
    # ===================================================================
    ranking_methods = ["total", "outgoing", "causal"]
    ablation_modes = [
        ("all-positions", False),
        ("last-token-only", True),
    ]

    # Collect results for summary table
    summary = {}  # (method, mode) -> accuracy

    for method in ranking_methods:
        circuit = select_top_features(scores, method, TOP_K_FEATURES)
        total_feats = sum(len(v) for v in circuit.values())

        for mode_label, last_token_only in ablation_modes:
            key = (method, mode_label)
            print(f"\n{'─' * 70}")
            print(f"ABLATION: rank={method}, mode={mode_label}")
            print(f"  Features: top {TOP_K_FEATURES}/layer, {total_feats} total across {len(circuit)} layers")
            print(f"{'─' * 70}")

            acc, results = run_ablation_experiment(
                key, circuit, saes, model, tokenizer, eval_prompts, device,
                last_token_only=last_token_only,
            )
            summary[key] = acc
            drop = baseline_acc - acc
            print(f"Accuracy: {acc:.1%}  (drop: {drop:+.1%})")

            errors = [r for r in results if not r["correct"]]
            if errors and acc < 0.5:
                print(f"Sample errors:")
                for r in errors[:5]:
                    print(f"  {r['prompt']} expected={r['answer']}, got={r['predicted']} (raw='{r['raw']}')")

    # ===================================================================
    # RANDOM ABLATION CONTROLS (both modes)
    # ===================================================================
    rng = random.Random(99)
    # Use the "total" ranking to determine which layers and how many features
    circuit_total = select_top_features(scores, "total", TOP_K_FEATURES)

    for mode_label, last_token_only in ablation_modes:
        print(f"\n{'─' * 70}")
        print(f"RANDOM CONTROL: mode={mode_label} ({N_RANDOM_TRIALS} trials)")
        print(f"{'─' * 70}")

        random_accs = []
        for trial in range(N_RANDOM_TRIALS):
            random_circuit = {}
            for layer_idx, feat_ids in circuit_total.items():
                n_feats = len(feat_ids)
                circuit_set = set(feat_ids)
                candidates = [i for i in range(16384) if i not in circuit_set]
                random_circuit[layer_idx] = rng.sample(candidates, n_feats)

            acc, _ = run_ablation_experiment(
                f"random-{trial}", random_circuit, saes,
                model, tokenizer, eval_prompts, device,
                last_token_only=last_token_only,
            )
            random_accs.append(acc)
            print(f"  Trial {trial + 1}: {acc:.1%}")

        mean_acc = sum(random_accs) / len(random_accs)
        summary[("random", mode_label)] = mean_acc
        print(f"  Mean: {mean_acc:.1%}  (drop: {baseline_acc - mean_acc:+.1%})")

    # ===================================================================
    # PER-LAYER ABLATION (last-token-only, causal ranking)
    # ===================================================================
    print(f"\n{'─' * 70}")
    print("PER-LAYER ABLATION (last-token-only, causal ranking)")
    print(f"{'─' * 70}")

    circuit_causal = select_top_features(scores, "causal", TOP_K_FEATURES)
    layer_impacts = []
    for layer_idx in sorted(circuit_causal.keys()):
        single = {layer_idx: circuit_causal[layer_idx]}
        acc, _ = run_ablation_experiment(
            f"L{layer_idx}", single, saes,
            model, tokenizer, eval_prompts, device,
            last_token_only=True,
        )
        drop = baseline_acc - acc
        layer_impacts.append((layer_idx, acc, drop))

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nBaseline: {baseline_acc:.1%}")
    print()
    print(f"{'Ranking':<12} {'All-pos':>10} {'Last-tok':>10}")
    print("─" * 34)
    for method in ranking_methods:
        all_acc = summary.get((method, "all-positions"), None)
        last_acc = summary.get((method, "last-token-only"), None)
        all_str = f"{all_acc:.1%}" if all_acc is not None else "—"
        last_str = f"{last_acc:.1%}" if last_acc is not None else "—"
        print(f"{method:<12} {all_str:>10} {last_str:>10}")
    rand_all = summary.get(("random", "all-positions"), None)
    rand_last = summary.get(("random", "last-token-only"), None)
    rand_all_str = f"{rand_all:.1%}" if rand_all is not None else "—"
    rand_last_str = f"{rand_last:.1%}" if rand_last is not None else "—"
    print(f"{'random':<12} {rand_all_str:>10} {rand_last_str:>10}")

    print(f"\nPer-layer impact (last-token-only, causal ranking, sorted by drop):")
    layer_impacts.sort(key=lambda x: x[2], reverse=True)
    for layer_idx, acc, drop in layer_impacts:
        bar = "█" * int(drop * 100) if drop > 0 else ""
        print(f"  L{layer_idx:2d}: {acc:.1%} (drop: {drop:+.1%}) {bar}")

    # Cleanup
    del model, tokenizer
    _sae_cache.clear()
    gc.collect()
