"""Verify: does find_frequent_nonzero_indices return feature indices or seq positions?"""
import torch

# Simulate: batch_size=10, seq_len=11, d_sae=16384
# (matching the actual experiment shapes)
batch_size, seq_len, d_sae = 10, 11, 16384

# Create a sparse tensor with some features active
tensor = torch.zeros(batch_size, seq_len, d_sae)
# Make a few features active across most/all prompts at various positions
for b in range(batch_size):
    tensor[b, 3, 100] = 1.0   # feature 100 at position 3
    tensor[b, 5, 200] = 1.0   # feature 200 at position 5
    tensor[b, 7, 5000] = 1.0  # feature 5000 at position 7

# --- Current (buggy) implementation ---
non_zero_count = torch.sum(tensor != 0, dim=0) / tensor.shape[0]
print(f"non_zero_count shape: {non_zero_count.shape}")  # (seq_len, d_sae)

result = torch.where(non_zero_count >= 0.9)
print(f"torch.where returns {len(result)} tensors")
print(f"  [0] (seq positions): {result[0]}")   # should be [3, 5, 7]
print(f"  [1] (feature indices): {result[1]}")  # should be [100, 200, 5000]

valid_indices = result[0]  # BUG: this is seq positions, not feature indices!
print(f"\nBUGGY result (used as feature indices): {valid_indices}")
print(f"  These are seq positions, not feature indices!")

# --- Fixed implementation ---
# Aggregate over seq dim first: a feature is "active" if nonzero at ANY position
any_active = (tensor != 0).any(dim=1)  # (batch_size, d_sae)
freq = any_active.float().mean(dim=0)  # (d_sae,)
correct_indices = torch.where(freq >= 0.9)[0]
print(f"\nFIXED result (actual feature indices): {correct_indices}")
print(f"  These are the real SAE feature indices: {correct_indices.tolist()}")
