import gc

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from feature_circuit_discovery.data import canonical_sae_filenames


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
        repo_id="google/gemma-scope-2b-pt-res",
        filename=filename,
        force_download=False,
    )
    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}
    sae = JumpReLUSAE(params["W_enc"].shape[0], params["W_enc"].shape[1])
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
        # Handle the case where output is a tuple
        if isinstance(output, tuple):
            # Typically, the first element is the tensor we need
            output_tensor = output[0]
        else:
            output_tensor = output

        # Ensure the output_tensor has at least 2 dimensions
        if output_tensor.dim() < 2:
            raise ValueError(f"Expected output tensor to have at least 2 dimensions, got {output_tensor.dim()}.")

        # Select the last token in the sequence
        last_token_activation = output_tensor[:, -1, :]  # Shape: (batch_size, hidden_size)
        activations["output"] = last_token_activation

    # Register the forward hook on the target layer
    handle = model.model.layers[target_layer].register_forward_hook(hook_fn)
    
    try:
        # Forward pass to trigger the hook
        model(inputs)
    finally:
        # Ensure the hook is removed even if an error occurs
        handle.remove()

    return activations["output"]

def get_active_features(tokenized_prompts, model, device, threshold=0.9):
    feature_acts = []

    for layer in tqdm(range(model.config.num_hidden_layers)):
        sae = load_sae(canonical_sae_filenames[layer], device)
        activations = gather_residual_activations(model, layer, tokenized_prompts)
        sae_activations = sae.encode(activations)
        # batch, seq, d_sae
        feature_acts.append(sae_activations)

    # channge this to finding the features that activate in most/all of the forward
    # passes in rather than looking at all of the activated ones

    # find all the activated features in all the layers
    print(feature_acts[0].shape)
    activated_features = find_frequent_nonzero_indices(feature_acts, threshold)

    return activated_features

def get_active_features_in_layer(tokenized_prompts, model, layer, threshold=0.9):
    device = model.device
    sae = load_sae(canonical_sae_filenames[layer], device)
    activations = gather_residual_activations(model, layer, tokenized_prompts)
    sae_activations = sae.encode(activations)
    activated_features = find_frequent_nonzero_indices([sae_activations], threshold)
    return activated_features[0]



def describe_feature(layer, idx):
    """
    Returns a string description of a a feature of a certain layer and index of gemma 2 2b
    """
    pass


def describe_features(layers, indices):
    """
    Describes multiple features of multiplae layers. layers and indices should therefore be 1 and 2 dimensional lists of integers.
    """
    pass


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



def model_with_added_feature_vector(model, inputs, sae, layer_l, feature_idx, scalar, layer_d):
    """
    Modifies the model's forward pass by adding a scaled feature vector to the residual stream at a specified layer.

    Args:
        model: The transformer model.
        inputs: The inputs to the model (can be a batch).
        sae: The Sparse Autoencoder (SAE) object.
        layer_l: The layer at which to add the feature vector.
        feature_idx: The index of the feature in the SAE.
        scalar: The scalar to scale the feature vector before adding.
        layer_d: The downstream layer from which to extract residual activations.

    Returns:
        logits: The output logits from the forward pass.
        downstream_act: The residual stream activations at layer_d.
    """
    # Ensure the feature vector is on the correct device and detached
    feature_vector = sae.W_dec[feature_idx, :].to(inputs.device).detach()  # Shape: (d_model,)
    feature_vector = feature_vector.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, d_model)
    
    # Define hooks
    downstream_act = {}

    def add_feature_vector_hook(module, module_input, module_output):
        # Add the scaled feature vector in-place to save memory
        module_output[0].add_(scalar * feature_vector)
        return module_output

    def gather_downstream_act_hook(module, module_input, module_output):
        # Detach the activations to prevent graph retention
        downstream_act['act'] = module_output[0].detach()
        return module_output

    # Register hooks
    handle_add = model.model.layers[layer_l].register_forward_hook(add_feature_vector_hook)
    handle_gather = model.model.layers[layer_d].register_forward_hook(gather_downstream_act_hook)

    # Run the forward pass within no_grad to disable gradient tracking
    with torch.no_grad():
        outputs = model(inputs)

    # Remove hooks immediately after the forward pass
    handle_add.remove()
    handle_gather.remove()

    # Detach logits to free memory
    logits = outputs.logits.detach()

    # Retrieve the downstream activations
    downstream_act = downstream_act.get('act', None)

    return logits, downstream_act