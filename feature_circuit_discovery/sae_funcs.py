import gc

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm

from feature_circuit_discovery.data import (
    MAX_SAE_LAYER,
    NEURONPEDIA_MODEL_ID,
    NEURONPEDIA_SAE_SET,
    SAE_HF_REPO,
    canonical_sae_filenames,
)


class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold).float()
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon


def load_sae(filename, device):
    path_to_params = hf_hub_download(
        repo_id=SAE_HF_REPO,
        filename=filename,
        force_download=False,
    )
    tensors = load_safetensors(path_to_params)
    # Gemma Scope 2 uses lowercase keys (w_enc, b_dec); remap to JumpReLUSAE (W_enc, b_dec)
    KEY_MAP = {"w_enc": "W_enc", "w_dec": "W_dec", "b_enc": "b_enc", "b_dec": "b_dec", "threshold": "threshold"}
    pt_params = {KEY_MAP.get(k, k): v for k, v in tensors.items()}
    pt_params = {k: v.to(device) for k, v in pt_params.items()}
    sae = JumpReLUSAE(pt_params["W_enc"].shape[0], pt_params["W_enc"].shape[1])
    sae.load_state_dict(pt_params)
    return sae.to(device)


def find_frequent_nonzero_indices(tensor_list, threshold=0.9):
    result_indices = []

    for tensor in tensor_list:
        non_zero_count = torch.sum(tensor != 0, dim=0) / tensor.shape[0]
        valid_indices = torch.where(non_zero_count >= threshold)[
            0
        ]  # Find the indices where the count of non-zero entries meets the threshold
        result_indices.append(valid_indices)  # Append these indices to the result list
    return result_indices


def gather_residual_activations(model, target_layer, inputs):
    activations = {}

    def hook_fn(module, input, output):
        # Gemma 2 layers return tuple (hidden_states, ...); Gemma 3 returns tensor directly
        activations["output"] = output[0] if isinstance(output, tuple) else output

    handle = model.model.layers[target_layer].register_forward_hook(hook_fn)
    model(inputs)
    handle.remove()

    return activations["output"]


def get_active_features(prompts, tokenizer, model, device, threshold=0.9):
    # change this to tokenizer().[whatever attrivutes of the return object we need]
    tokenized_prompts = (
        tokenizer(prompts, return_tensors="pt", add_special_tokens=True, padding=True)
        .data["input_ids"]
        .to(device)
    )
    feature_acts = []

    num_layers = min(model.config.num_hidden_layers, MAX_SAE_LAYER + 1)
    for layer in tqdm(range(num_layers)):
        sae = load_sae(canonical_sae_filenames[layer], device)
        activations = gather_residual_activations(model, layer, tokenized_prompts)
        sae_activations = sae.encode(activations)
        feature_acts.append(sae_activations)

    # channge this to finding the features that activate in most/all of the forward
    # passes in rather than looking at all of the activated ones

    # find all the activated features in all the layers
    activated_features = find_frequent_nonzero_indices(feature_acts, threshold=0.9)

    return activated_features


def describe_feature(layer: int, idx: int) -> str | None:
    """Fetch the auto-interp description of an SAE feature from Neuronpedia.

    Args:
        layer: Layer index (0-20 for Gemma 3 1B resid_post_all).
        idx: Feature index in the 16k-width SAE.

    Returns:
        Description string, or None if unavailable.
    """
    import requests

    url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL_ID}/{layer}-{NEURONPEDIA_SAE_SET}/{idx}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        explanations = data.get("explanations", [])
        if explanations:
            return explanations[0].get("description")
        return None
    except Exception:
        return None


def describe_features(
    features: list[tuple[int, int]],
    max_concurrent: int = 10,
) -> dict[tuple[int, int], str]:
    """Fetch descriptions for multiple features from Neuronpedia.

    Args:
        features: List of (layer, feature_idx) tuples.
        max_concurrent: Max concurrent requests.

    Returns:
        Dict mapping (layer, feature_idx) -> description string.
    """
    import concurrent.futures
    import requests

    results: dict[tuple[int, int], str] = {}

    def fetch_one(layer: int, idx: int) -> tuple[int, int, str | None]:
        url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL_ID}/{layer}-{NEURONPEDIA_SAE_SET}/{idx}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return (layer, idx, None)
            data = resp.json()
            explanations = data.get("explanations", [])
            if explanations:
                return (layer, idx, explanations[0].get("description"))
            return (layer, idx, None)
        except Exception:
            return (layer, idx, None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(fetch_one, layer, idx) for layer, idx in features]
        for future in concurrent.futures.as_completed(futures):
            layer, idx, desc = future.result()
            if desc:
                results[(layer, idx)] = desc

    return results


def compute_gradient_matrix(
    inputs,
    upstream_layer_idx,
    downstream_layer_idx,
    upstream_features,
    downstream_features,
    model,
    verbose=False,
):
    device = model.device

    if model is None:
        raise ValueError("model must be provided")

    if device is None:
        device = next(model.parameters()).device

    # Clear GPU cache and enable gradient computation
    torch.cuda.empty_cache()
    gc.collect()
    torch.set_grad_enabled(True)

    if verbose:
        print()
        print(f"loading upstream sae (Layer {upstream_layer_idx})")
    sae_upstream = load_sae(canonical_sae_filenames[upstream_layer_idx], device)

    if verbose:
        print("upsteam sae loaded")
        print(f"loading downstram sae (Layer {downstream_layer_idx})")
    sae_downstream = load_sae(canonical_sae_filenames[downstream_layer_idx], device)

    if verbose:
        print("downsteam sae loaded")
    m = len(upstream_features)
    n = len(downstream_features)

    d_model = sae_upstream.W_dec.size(1)

    # Define the scalar parameters 'a' as a tensor with requires_grad=True
    a = torch.zeros(m, requires_grad=True, device=device)  # Shape: (m,)

    feature_vectors = []
    if verbose:
        print("getting upstram feature vectors")
    for k in upstream_features:
        feature_vector = sae_upstream.W_dec[k, :]
        feature_vectors.append(feature_vector)

    feature_vectors = torch.stack(feature_vectors)

    def modify_residual_activations(module, input, output):
        residual = output[0] if isinstance(output, tuple) else output
        residual_modified = residual.clone()

        weighted_feature_vectors = (
            a.view(m, 1, 1) * feature_vectors.view(m, 1, d_model)
        ).sum(dim=0)
        expanded_weighted_feature_vector = weighted_feature_vectors.expand(
            residual_modified.size(0), -1, -1
        )

        residual_modified[:, 0:1, :] += expanded_weighted_feature_vector

        return (
            (residual_modified,) + output[1:]
            if isinstance(output, tuple)
            else residual_modified
        )

    if verbose:
        print("conducting forward pass")
    hook = model.model.layers[upstream_layer_idx].register_forward_hook(
        modify_residual_activations
    )
    outputs = model(inputs, output_hidden_states=True)

    hook.remove()

    hidden_states = outputs.hidden_states
    act_upstream = hidden_states[upstream_layer_idx]
    act_downstream = hidden_states[downstream_layer_idx]

    sae_downstream_acts = sae_downstream.encode(act_downstream)

    batch_idx = 0
    seq_idx = 0

    features_downstream = sae_downstream_acts[batch_idx, seq_idx, downstream_features]

    gradient_matrix = torch.zeros(n, m, device=device)

    # Compute gradients for each feature in the downstream layer with respect to 'a'
    if verbose:
        print("computing gradients")
        for i in tqdm(range(n)):
            if a.grad is not None:
                a.grad.zero_()
            features_downstream[i].backward(retain_graph=True)
            gradient_matrix[i, :] = a.grad.detach()
            a.grad.zero_()
    else:
        for i in range(n):
            if a.grad is not None:
                a.grad.zero_()
            features_downstream[i].backward(retain_graph=True)
            gradient_matrix[i, :] = a.grad.detach()
            a.grad.zero_()
    if verbose:
        print("clean up")
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    if verbose:
        print("done")
    return gradient_matrix
