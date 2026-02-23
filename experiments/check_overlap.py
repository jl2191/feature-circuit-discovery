"""Check if batch-selected features overlap with actually-active features for single prompt."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.feature_grad_exp_optimized import load_sae, get_active_features
from feature_circuit_discovery.data import canonical_sae_filenames

device = torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b", dtype=torch.bfloat16
).to(device)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

prompts = ["754 + 214 = ", "125 + 859 = "]
inputs = tokenizer.encode(prompts[0], return_tensors="pt", add_special_tokens=True).to(device)

print(f"Input tokens: {tokenizer.convert_ids_to_tokens(inputs[0])}")
print(f"Input shape: {inputs.shape}")

# Get batch-selected features
print("\n1. Getting batch-selected features (threshold=0.9)...")
activated_features = get_active_features(prompts, tokenizer, model, device)

# Now check overlap at each position for a single prompt
print("\n2. Checking overlap with single-prompt activations...")
with torch.no_grad():
    outputs = model(inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

for layer_idx in [0, 1, 2]:
    sae = load_sae(canonical_sae_filenames[layer_idx], device)
    act = sae.encode(hidden_states[layer_idx + 1].float())  # (1, seq_len, d_sae)

    batch_feats = set(activated_features[layer_idx].tolist())
    print(f"\n  Layer {layer_idx}: {len(batch_feats)} batch-selected features")

    for seq_idx in range(act.shape[1]):
        token = tokenizer.convert_ids_to_tokens([inputs[0, seq_idx]])[0]
        active_at_pos = set(torch.where(act[0, seq_idx] != 0)[0].tolist())
        overlap = batch_feats & active_at_pos
        print(f"    seq={seq_idx} ('{token}'): {len(active_at_pos)} active, "
              f"{len(overlap)} overlap with batch-selected")

    # Check last position specifically
    last_pos = act.shape[1] - 1
    active_last = set(torch.where(act[0, last_pos] != 0)[0].tolist())
    overlap_last = batch_feats & active_last
    print(f"    >>> Last position overlap: {len(overlap_last)}/{len(batch_feats)} "
          f"({100*len(overlap_last)/max(len(batch_feats),1):.1f}%)")

    if overlap_last:
        print(f"    >>> Overlapping feature indices (first 10): {sorted(overlap_last)[:10]}")

    # Also check: what fraction of batch-selected features are active at ANY position?
    any_pos_active = set()
    for s in range(act.shape[1]):
        any_pos_active |= set(torch.where(act[0, s] != 0)[0].tolist())
    overlap_any = batch_feats & any_pos_active
    print(f"    >>> Any-position overlap: {len(overlap_any)}/{len(batch_feats)} "
          f"({100*len(overlap_any)/max(len(batch_feats),1):.1f}%)")
