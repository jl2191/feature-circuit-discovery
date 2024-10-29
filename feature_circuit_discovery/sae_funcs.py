# %%
import gc
import http.client
import json
import os

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


def find_frequent_nonzero_indices(tensor_list, threshold=0.9, activation_threshold = 0.0):
    """
    Identifies indices with non-zero values meeting or exceeding a threshold.

    Args:
        tensor_list: A list of tensors to analyze.
        threshold: The minimum frequency (as a fraction of the total) of non-zero
            values required for an index to be included. Defaults to 0.9.

    Returns:
        A list of tensors with indices meeting the threshold for each input tensor.
    """
    result_indices = []

    for tensor in tensor_list:
        non_zero_count = torch.sum(tensor > activation_threshold, dim=0) / tensor.shape[0]
        valid_indices = torch.where(non_zero_count >= threshold)[0]
        result_indices.append(valid_indices)
    return result_indices


def gather_residual_activations(model, target_layer, inputs):
    """
    Gathers residual activations from a model layer for the last token.

    Args:
        model: The model from which to gather activations.
        target_layer: The index of the layer from which to gather activations.
        inputs: The inputs to the model.

    Returns:
        A tensor with the last token activations from the specified layer.
    """
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
            raise ValueError(
                f"Expected output tensor to have at least 2 dimensions, got {output_tensor.dim()}."
            )

        last_token_activation = output_tensor[:, -1, :]
        # shape: (batch_size, hidden_size)
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


def get_active_features(tokenized_prompts, model, device, verbose=False, threshold=0.9, activation_threshold = 0.0):
    """
    Identifies active features in the model's layers based on tokenized prompts.

    Args:
        tokenized_prompts: The tokenized input prompts.
        model: The model from which to gather activations.
        device: The device on which the model is running.
        verbose: Whether to print additional information during processing. Defaults
            to False.
        threshold: The threshold for determining active features. Defaults to 0.9.

    Returns:
        A dictionary with layer indices as keys and lists of active features as values.
    """
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
    if verbose:
        print(feature_acts[0].shape)
    activated_features = find_frequent_nonzero_indices(feature_acts, threshold, activation_threshold = activation_threshold)

    return {layer: features for layer, features in enumerate(activated_features)}


def get_active_features_in_layer(tokenized_prompts, model, layer, threshold=0.9):
    """
    Identifies active features in a specific layer of the model based on tokenized
    prompts.

    Args:
        tokenized_prompts: The tokenized input prompts.
        model: The model from which to gather activations.
        layer: The specific layer to analyze.
        threshold: The threshold for determining active features. Defaults to 0.9.

    Returns:
        A list of active features for the specified layer.
    """
    device = model.device
    sae = load_sae(canonical_sae_filenames[layer], device)
    activations = gather_residual_activations(model, layer, tokenized_prompts)
    sae_activations = sae.encode(activations)
    activated_features = find_frequent_nonzero_indices([sae_activations], threshold)
    return activated_features[0]


def describe_feature(
    feature_id, layer, sae_id=None, model_id="gemma-2-2b", verbose=False, mode="local"
):
    """
    Retrieves a description of a specific feature from a given model layer.

    The local mode assumes that the explanations have been downloaded and saved in the
    datasets/neuronpedia/explanations-only directory. For convenience, we have included
    the explanation files for gemma-2-2b gemmascope-res-16k explanations in the repo, as
    well as a download script for others.

    Use the describe_feature*s*() function for a lot of features.

    Remote calls require a Neuronpedia API key, see "https://www.neuronpedia.org/api-doc".

    Args:
        feature_id: The identifier of the feature within the layer.
        layer: The index of the layer containing the feature.
        sae_id: The identifier of the SAE. If None, defaults to '{layer}-gemmascope-res-16k'.
        model_id: The identifier of the model. Defaults to 'gemma-2-2b'.
        verbose: Whether to print additional information during the request. Defaults
            to False.
        mode: The mode of operation, either 'remote' or 'local'. Defaults to 'local'.

    Returns:
        A string description of the feature. If the description cannot be retrieved,
        returns an error message.
    """
    if sae_id is None:
        sae_id = f"{layer}-gemmascope-res-16k"

    if mode == "remote":
        conn = http.client.HTTPSConnection("www.neuronpedia.org")
        headers = {"X-Api-Key": os.getenv("NEURONPEDIA_API_KEY")}
        endpoint = f"/api/feature/{model_id}/{sae_id}/{feature_id}"

        if verbose:
            print(f"Requesting feature description from {endpoint}")

        conn.request("GET", endpoint, headers=headers)
        res = conn.getresponse()
        data = res.read()

        try:
            feature_data = json.loads(data.decode("utf-8"))
            explanations = feature_data.get("explanations", "No explanations available")
            if isinstance(explanations, list) and explanations:
                explanations = explanations[0]
            description = (
                explanations.get("description", "No description available")
                if isinstance(explanations, dict)
                else "No description available"
            )

            if verbose:
                print(
                    f"Feature description for {feature_id} in layer {layer}: {description}"
                )

            return description
        except json.JSONDecodeError:
            return "Error decoding the response."

    elif mode == "local":
        file_path = (
            f"/tmp/feature-circuit-discovery/datasets/neuronpedia/explanations/"
            f"{model_id}_{sae_id}.json"
        )
        print(file_path)
        try:
            with open(file_path, "r") as file:
                feature_data = json.load(file)
                explanations = next(
                    (item for item in feature_data if item["index"] == str(feature_id)),
                    {"description": "No description available"},
                )
                description = explanations.get(
                    "description", "No description available"
                )

                if verbose:
                    print(
                        f"Feature description for {feature_id} in layer {layer}: {description}"
                    )

                return description
        except FileNotFoundError:
            return f"File not found: {file_path}"
        except json.JSONDecodeError:
            return "Error decoding the file."

    else:
        return "Invalid mode. Choose either 'remote' or 'local'."


def describe_features(
    features_dict, sae_id=None, model_id="gemma-2-2b", verbose=False, mode="local"
):
    """
    Retrieves descriptions of specific features from given model layers.

    The local mode assumes that the explanations have been downloaded and saved in the
    datasets/neuronpedia/explanations-only directory. For convenience, we have included
    the explanation files for gemma-2-2b gemmascope-res-16k explanations in the repo, as
    well as a download script for others.

    Remote calls require a Neuronpedia API key, see "https://www.neuronpedia.org/api-doc".
    However, this will be quite slow if you have a lot of features and so local is
    recommended.

    Args:
        features_dict: A dictionary with layer numbers as keys and lists of feature
            indices as values.
        sae_id: The identifier of the SAE. If None, defaults to '{layer}-gemmascope-res-16k'.
        model_id: The identifier of the model. Defaults to 'gemma-2-2b'.
        verbose: Whether to print additional information during the request. Defaults
            to False.
        mode: The mode of operation, either 'remote' or 'local'. Defaults to 'local'.

    Returns:
        A dictionary with layer numbers as keys and lists of dictionaries containing
        feature indices and their descriptions as values.
    """
    descriptions = {}

    if mode == "remote":
        conn = http.client.HTTPSConnection("www.neuronpedia.org")
        headers = {"X-Api-Key": os.getenv("NEURONPEDIA_API_KEY")}

        for layer, feature_ids in features_dict.items():
            if sae_id is None:
                current_sae_id = f"{layer}-gemmascope-res-16k"
            else:
                current_sae_id = sae_id

            descriptions[layer] = []
            for feature_id in feature_ids:
                endpoint = f"/api/feature/{model_id}/{current_sae_id}/{feature_id}"

                if verbose:
                    print(f"Requesting feature description from {endpoint}")

                conn.request("GET", endpoint, headers=headers)
                res = conn.getresponse()
                data = res.read()

                try:
                    feature_data = json.loads(data.decode("utf-8"))
                    explanations = feature_data.get(
                        "explanations", "No explanations available"
                    )
                    if isinstance(explanations, list) and explanations:
                        explanations = explanations[0]
                    description = (
                        explanations.get("description", "No description available")
                        if isinstance(explanations, dict)
                        else "No description available"
                    )

                    if verbose:
                        print(
                            f"Feature description for {feature_id} in layer {layer}: {description}"
                        )

                    descriptions[layer].append(
                        {"feature_id": feature_id, "description": description}
                    )
                except json.JSONDecodeError:
                    descriptions[layer].append(
                        {
                            "feature_id": feature_id,
                            "description": "Error decoding the response.",
                        }
                    )

    elif mode == "local":
        for layer, feature_ids in features_dict.items():
            if sae_id is None:
                current_sae_id = f"{layer}-gemmascope-res-16k"
            else:
                current_sae_id = sae_id

            file_path = (
                f"/tmp/feature-circuit-discovery/datasets/neuronpedia/explanations/"
                f"{model_id}_{current_sae_id}.json"
            )
            descriptions[layer] = []
            try:
                with open(file_path, "r") as file:
                    feature_data = json.load(file)
                    for feature_id in feature_ids:
                        explanations = next(
                            (
                                item
                                for item in feature_data
                                if item["index"] == str(feature_id)
                            ),
                            {"description": "No description available"},
                        )
                        description = explanations.get(
                            "description", "No description available"
                        )

                        if verbose:
                            print(
                                f"Feature description for {feature_id} in layer "
                                f"{layer}: {description}"
                            )

                        descriptions[layer].append(
                            {"feature_id": feature_id, "description": description}
                        )
            except FileNotFoundError:
                for feature_id in feature_ids:
                    descriptions[layer].append(
                        {
                            "feature_id": feature_id,
                            "description": f"File not found: {file_path}",
                        }
                    )
            except json.JSONDecodeError:
                for feature_id in feature_ids:
                    descriptions[layer].append(
                        {
                            "feature_id": feature_id,
                            "description": "Error decoding the file.",
                        }
                    )

    else:
        return "Invalid mode. Choose either 'remote' or 'local'."

    return descriptions


def compute_gradient_matrix(
    inputs,
    upstream_layer_idx,
    downstream_layer_idx,
    upstream_features,
    downstream_features,
    model,
    verbose=False,
    sae_upstream = None,
    sae_downstream = None,
):
    """
    Computes the gradient matrix between upstream and downstream features in a model.

    Args:
        inputs: The tokenized input prompts.
        upstream_layer_idx: The index of the upstream layer.
        downstream_layer_idx: The index of the downstream layer.
        upstream_features: The list of active features in the upstream layer.
        downstream_features: The list of active features in the downstream layer.
        model: The model from which to gather activations.
        verbose: Whether to print verbose output. Defaults to False.

    Returns:
        A gradient matrix of shape (n, m) where n is the number of downstream features
        and m is the number of upstream features.
    """
    device = model.device

    if model is None:
        raise ValueError("model must be provided")

    if device is None:
        device = next(model.parameters()).device

    # Clear GPU cache and enable gradient computation
    torch.cuda.empty_cache()
    gc.collect()
    torch.set_grad_enabled(True)
    if sae_upstream == None:
        if verbose:
            print()
            print(f"loading upstream sae (Layer {upstream_layer_idx})")
        sae_upstream = load_sae(canonical_sae_filenames[upstream_layer_idx], device)

    if sae_downstream == None:
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
        feature_vector = sae_upstream.W_dec[k, :].clone().detach().requires_grad_(True)
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

        residual_modified[:, :, :] += expanded_weighted_feature_vector

        return (
            (residual_modified,) + output
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
    
    act_downstream = hidden_states[downstream_layer_idx]

    sae_downstream_acts = sae_downstream.encode(act_downstream)


    seq_idx = -1  # Last token
    features_downstream = sae_downstream_acts[:, seq_idx, downstream_features]  # Shape: (batch_size, n)

    gradient_matrix = torch.zeros(n, m, device=device)  # Shape: (n, m)

    if verbose:
        print("computing gradients for all batch elements at the last token")

    # Compute gradients for each downstream feature across all batch elements
    for i in range(n):
        if a.grad is not None:
            a.grad.zero_()
        # Sum over the batch dimension to get a scalar output
        scalar_output = features_downstream[:, i].sum()
        scalar_output.backward(retain_graph=True)
        gradient_matrix[i, :] = a.grad.detach()
        


    if verbose:
        print("clean up")
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    if verbose:
        print("done")
    return gradient_matrix


def model_with_added_feature_vector(
    model, inputs, sae, layer_l, feature_idx, scalar, layer_d
):
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
    feature_vector = (
        sae.W_dec[feature_idx, :].to(inputs.device).detach()
    )  # Shape: (d_model,)
    feature_vector = feature_vector.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, d_model)

    # Define hooks
    downstream_act = {}

    def add_feature_vector_hook(module, module_input, module_output):
        # Add the scaled feature vector in-place to save memory
        module_output[0].add_(scalar * feature_vector)
        return module_output

    def gather_downstream_act_hook(module, module_input, module_output):
        # Detach the activations to prevent graph retention
        downstream_act["act"] = module_output[0].detach()
        return module_output

    # Register hooks
    handle_add = model.model.layers[layer_l].register_forward_hook(
        add_feature_vector_hook
    )
    handle_gather = model.model.layers[layer_d].register_forward_hook(
        gather_downstream_act_hook
    )

    # Run the forward pass within no_grad to disable gradient tracking
    with torch.no_grad():
        outputs = model(inputs)

    # Remove hooks immediately after the forward pass
    handle_add.remove()
    handle_gather.remove()

    # Detach logits to free memory
    logits = outputs.logits.detach()

    # Retrieve the downstream activations
    downstream_act = downstream_act.get("act", None)

    return logits, downstream_act
