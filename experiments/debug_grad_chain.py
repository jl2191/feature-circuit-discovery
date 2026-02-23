"""
Debug: trace where the gradient chain breaks between `a` and downstream features.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feature_circuit_discovery.data import canonical_sae_filenames
from feature_circuit_discovery.sae_funcs import JumpReLUSAE, load_sae

device = torch.device("cpu")

# Use bfloat16 to save memory — we're just debugging the graph structure
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b", dtype=torch.bfloat16
).to(device)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

inputs = tokenizer.encode(
    "what's your favourite color?", return_tensors="pt", add_special_tokens=True
).to(device)

upstream_layer = 0
downstream_layer = 1

sae_up = load_sae(canonical_sae_filenames[upstream_layer], device)
sae_down = load_sae(canonical_sae_filenames[downstream_layer], device)

# Use just 3 upstream features for debugging
m = 3
upstream_features = torch.arange(m)
d_model = sae_up.W_dec.size(1)

a = torch.zeros(m, requires_grad=True, device=device)
feature_vectors = sae_up.W_dec[upstream_features, :]  # (m, d_model)

hook_called = False

def modify_hook(module, input, output):
    global hook_called
    hook_called = True
    print(f"  Hook called. output type: {type(output)}")
    if isinstance(output, tuple):
        print(f"  output is tuple of length {len(output)}")
        residual = output[0]
    else:
        print(f"  output is {type(output)}")
        residual = output

    print(f"  residual shape: {residual.shape}, dtype: {residual.dtype}, "
          f"requires_grad: {residual.requires_grad}")

    residual_modified = residual.clone()
    weighted = (a.view(m, 1, 1) * feature_vectors.view(m, 1, d_model)).sum(dim=0)
    expanded = weighted.expand(residual_modified.size(0), -1, -1)

    print(f"  expanded dtype: {expanded.dtype}, requires_grad: {expanded.requires_grad}")

    # Check if dtypes match
    if residual_modified.dtype != expanded.dtype:
        print(f"  WARNING: dtype mismatch! residual={residual_modified.dtype}, expanded={expanded.dtype}")
        expanded = expanded.to(residual_modified.dtype)

    residual_modified[:, 0:1, :] += expanded

    print(f"  residual_modified requires_grad: {residual_modified.requires_grad}")

    if isinstance(output, tuple):
        return (residual_modified,) + output[1:]
    return residual_modified

print("1. Running forward pass with hook...")
hook = model.model.layers[upstream_layer].register_forward_hook(modify_hook)

with torch.enable_grad():
    outputs = model(inputs, output_hidden_states=True)

hook.remove()

hidden_states = outputs.hidden_states
print(f"\n2. Hidden states: {len(hidden_states)} tensors")
for i in range(min(4, len(hidden_states))):
    hs = hidden_states[i]
    print(f"  hidden_states[{i}]: shape={hs.shape}, dtype={hs.dtype}, "
          f"requires_grad={hs.requires_grad}, grad_fn={hs.grad_fn is not None}")

act_downstream = hidden_states[downstream_layer + 1]
print(f"\n3. Downstream hidden state (layer {downstream_layer} output = index {downstream_layer+1}):")
print(f"  requires_grad={act_downstream.requires_grad}, grad_fn={act_downstream.grad_fn}")

# Also check the raw downstream layer output (index = downstream_layer, not +1)
act_downstream_alt = hidden_states[downstream_layer]
print(f"\n3b. hidden_states[{downstream_layer}] (might be upstream layer output):")
print(f"  requires_grad={act_downstream_alt.requires_grad}, grad_fn={act_downstream_alt.grad_fn}")

print(f"\n4. Encoding with downstream SAE...")
sae_acts = sae_down.encode(act_downstream.float())
print(f"  sae_acts shape: {sae_acts.shape}, requires_grad: {sae_acts.requires_grad}, "
      f"grad_fn: {sae_acts.grad_fn}")

if sae_acts.requires_grad:
    # Pick a non-zero feature
    nonzero_mask = sae_acts[0, 0, :] != 0
    nonzero_indices = torch.where(nonzero_mask)[0]
    print(f"  Non-zero features at [0,0]: {len(nonzero_indices)}")
    if len(nonzero_indices) > 0:
        idx = nonzero_indices[0].item()
        target = sae_acts[0, 0, idx]
        print(f"\n5. Computing gradient of feature {idx} (value={target.item():.4f}) w.r.t. a...")
        grad = torch.autograd.grad(target, a, allow_unused=True)[0]
        print(f"  grad: {grad}")
    else:
        print("  No non-zero features found!")
else:
    print("  sae_acts doesn't require grad — chain is broken before SAE!")
    # Try to find where it breaks
    print(f"\n5. Trying gradient of raw hidden state w.r.t. a...")
    target = act_downstream[0, 0, 0]
    print(f"  target requires_grad: {target.requires_grad}")
    if target.requires_grad:
        grad = torch.autograd.grad(target, a, allow_unused=True)[0]
        print(f"  grad: {grad}")
    else:
        print("  Hidden state itself has no grad — model forward pass broke the chain!")
