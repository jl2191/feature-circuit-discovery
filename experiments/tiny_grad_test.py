"""Tiny test: just 5 features to quickly confirm non-zero gradients after bug fixes."""
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

prompts = ["754 + 214 = ", "125 + 859 = "]
inputs = tokenizer.encode(prompts[0], return_tensors="pt", add_special_tokens=True).to(device)

print("Getting active features...")
activated_features = get_active_features(prompts, tokenizer, model, device)

print(f"Features per layer (first 5): {[len(f) for f in activated_features[:5]]}")
print(f"Sample feature indices layer 0: {activated_features[0][:10]}")
print(f"Sample feature indices layer 1: {activated_features[1][:10]}")
print(f"Max feature idx layer 0: {activated_features[0].max() if len(activated_features[0]) > 0 else 'N/A'}")
print(f"Max feature idx layer 1: {activated_features[1].max() if len(activated_features[1]) > 0 else 'N/A'}")

# Take only first 5 features per layer for speed
up_feats = activated_features[0][:5]
down_feats = activated_features[1][:5]
print(f"\nTesting with {len(up_feats)} upstream, {len(down_feats)} downstream features")
print(f"  upstream indices: {up_feats}")
print(f"  downstream indices: {down_feats}")

grad_matrix = compute_gradient_matrix(
    inputs, 0, 1, up_feats, down_feats, model, verbose=True,
)
print(f"\nResult: shape={grad_matrix.shape}")
print(f"  max={grad_matrix.abs().max():.6f}, mean={grad_matrix.abs().mean():.6f}")
print(f"  nonzero={grad_matrix.count_nonzero()}/{grad_matrix.numel()}")
print(f"  matrix:\n{grad_matrix}")
