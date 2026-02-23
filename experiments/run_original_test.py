"""
Minimal test: run the ORIGINAL sae_funcs code in float32 on CPU
to check if gradients are non-zero (vs bfloat16 in the optimized version).
Only runs 2 layer pairs to save time.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feature_circuit_discovery.sae_funcs import (
    compute_gradient_matrix,
    get_active_features,
)

device = torch.device("cpu")

# Load in float32 (same as original, but without device_map="auto")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b").to(device)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# Same prompts as original
prompt_data = [
    "what's your favourite color?",
    "who should be president?",
    "why don't you like me?",
]

inputs = tokenizer.encode(
    prompt_data[0], return_tensors="pt", add_special_tokens=True
).to(device)

print("Getting active features...")
activated_features = get_active_features(prompt_data, tokenizer, model, device)
print(f"Features per layer (first 5): {[len(f) for f in activated_features[:5]]}")

# Only test 2 layer pairs to save time
for layer in range(2):
    print(f"\n--- Layer {layer} -> {layer+1} ---")
    grad_matrix = compute_gradient_matrix(
        inputs,
        layer,
        layer + 1,
        activated_features[layer],
        activated_features[layer + 1],
        model,
        verbose=True,
    )
    print(f"  shape={grad_matrix.shape}, max={grad_matrix.abs().max():.6f}, "
          f"mean={grad_matrix.abs().mean():.6f}, nonzero={grad_matrix.count_nonzero()}")
