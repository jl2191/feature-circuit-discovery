"""Debug: check if downstream features are actually active at seq_idx=0."""
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

print("\nGetting active features from batch...")
activated_features = get_active_features(prompts, tokenizer, model, device)
print(f"Layer 1 activated features: {len(activated_features[1])} features")

# Now run a single forward pass and check which features are active at seq_idx=0
with torch.no_grad():
    outputs = model(inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

sae_down = load_sae(canonical_sae_filenames[1], device)
act = sae_down.encode(hidden_states[2].float())  # layer 1 output

print(f"\nSAE activations shape: {act.shape}")  # [1, seq_len, 16384]
for seq_idx in range(act.shape[1]):
    token = tokenizer.convert_ids_to_tokens([inputs[0, seq_idx]])[0]
    total_active = (act[0, seq_idx, :] != 0).sum().item()
    # Check how many of the "activated_features" are actually active at this position
    feats_at_pos = act[0, seq_idx, activated_features[1]]
    active_count = (feats_at_pos != 0).sum().item()
    print(f"  seq_idx={seq_idx} ('{token}'): {total_active} total active, "
          f"{active_count}/{len(activated_features[1])} of selected features active")
