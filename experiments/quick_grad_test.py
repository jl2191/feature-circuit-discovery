"""Quick test: verify the fixes produce non-zero gradients."""
import random
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.feature_grad_exp_optimized import (
    compute_gradient_matrix,
    get_active_features,
)

device = torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b", dtype=torch.bfloat16
).to(device)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

random.seed(42)
prompts = [f"{random.randint(100,999)} + {random.randint(100,999)} = " for _ in range(10)]
inputs = tokenizer.encode(prompts[0], return_tensors="pt", add_special_tokens=True).to(device)

print("Getting active features...")
activated_features = get_active_features(prompts, tokenizer, model, device)
print(f"Features per layer (first 5): {[len(f) for f in activated_features[:5]]}")

for layer in range(2):
    print(f"\n--- Layer {layer} -> {layer+1} ---")
    grad_matrix = compute_gradient_matrix(
        inputs, layer, layer + 1,
        activated_features[layer], activated_features[layer + 1],
        model, verbose=True,
    )
    print(f"  shape={grad_matrix.shape}")
    print(f"  max={grad_matrix.abs().max():.6f}, mean={grad_matrix.abs().mean():.6f}")
    print(f"  nonzero={grad_matrix.count_nonzero()}/{grad_matrix.numel()}")
