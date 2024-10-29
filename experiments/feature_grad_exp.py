# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from matplotlib.widgets import Slider
from transformers import AutoModelForCausalLM, AutoTokenizer
from feature_circuit_discovery.data import canonical_sae_filenames

from feature_circuit_discovery.sae_funcs import (
    compute_gradient_matrix,
    get_active_features,
    describe_features,
    load_sae
)

# %%
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map="auto",
)

device = model.device


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")


# %%
prompt_data = [
    "what's your favourite color?",
    "who should be president?",
    "why don't you like me?",
]

# %%
with open(
    "/tmp/feature-circuit-discovery/datasets/ioi/ioi_test_100.json", "rb"
) as file:
    prompt_data = json.load(file)
# %%
import random
#addition data
addition_prompts = []
for i in range(15):
    addition_prompts.append(f"{random.randint(0, 100)} + {random.randint(0, 100)} = ")

print(addition_prompts)
# %%
tokenized_prompts = (
    tokenizer(
        addition_prompts,#prompt_data["prompts"][15:30],
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    )
    .data["input_ids"]
    .to(device)
)

matrices = []

activated_features = get_active_features(tokenized_prompts, model, device, threshold=1, activation_threshold = 0)

for i in range(len(activated_features)):
    print()
    print(activated_features[i])
    print("length:", len(activated_features[i]))

print(activated_features)

# %%
"""
for layer in tqdm(range(len(activated_features) - 2)):
    grad_matrix = compute_gradient_matrix(
        tokenized_prompts,
        layer,
        layer + 1,
        activated_features[layer],
        activated_features[layer + 2],
        model,
        verbose=True,
    )
    matrices.append(grad_matrix)
"""
# %%
"""
from feature_circuit_discovery.graphs import draw_bipartite_graph

for i in matrices:
    draw_bipartite_graph(i, threshold=0)
"""
# %%
"""
for i in matrices:
    plt.imshow(i.cpu())
    plt.colorbar()
    plt.show()
"""    
# %%
"""
for i in matrices:
    print(i)"""

# %%
import gc
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
    """
    # Compute gradients for each downstream feature across all batch elements
    for i in range(n):
        if a.grad is not None:
            a.grad.zero_()
        # Sum over the batch dimension to get a scalar output
        scalar_output = features_downstream[:, i].sum()
        scalar_output.backward(retain_graph=True)
        gradient_matrix[i, :] = a.grad.detach()
    """
    if a.grad is not None:
        a.grad.zero_()
    scalar_outputs = features_downstream.sum(dim=0)  # Shape: (n,)
    print(scalar_outputs.shape)
    scalar_outputs.backward(gradient=torch.eye(n, device=device), retain_graph=True)
    gradient_matrix = a.grad.detach().t()  # Shape: (n, m)


    if verbose:
        print("clean up")
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    if verbose:
        print("done")
    return gradient_matrix


connections = {}
num_layers = len(activated_features)

for i in tqdm(range(num_layers)):
    upstream_sae = load_sae(canonical_sae_filenames[i], device)
    for j in range(i + 1, num_layers):
        # Compute the gradient matrix between layer i and layer j
        grad_matrix = compute_gradient_matrix(
            tokenized_prompts,
            i,
            j,
            activated_features[i],
            activated_features[j],
            model,
            #verbose=True,
            sae_upstream=upstream_sae
        )
        # Store the computed matrix in the dictionary
        connections[(i, j)] = grad_matrix


# %%
m_a = -10000.0
for i in connections.values():
    m = torch.max(torch.abs(i))
    if m>m_a:
        m_a = m


# %%

def plot_layer_connections(
    connections_dict,
    num_nodes_per_layer,
    window_width=10,
    window_height=10,
    margin=1,
    node_size=50,
    node_color="white",
    connection_color="black",
    top_n=30,  # Number of top nodes to label
    describe_features_func=None,  # Function to get descriptions
    text_bg_color="white",  # Background color for text labels
    text_bg_alpha=0.7,  # Transparency for text background
):
    layer_count = len(num_nodes_per_layer)

    # Calculate available vertical space
    available_height = window_height - 2 * margin

    # Find the maximum number of nodes in any layer
    max_nodes_in_layer = max(num_nodes_per_layer)

    # Calculate the vertical spacing between nodes
    if max_nodes_in_layer > 1:
        node_spacing = available_height / (max_nodes_in_layer - 1)
    else:
        node_spacing = 0  # If only one node, no spacing needed

    # Assign positions to each node
    node_positions = {}
    for i, nodes_in_layer in enumerate(num_nodes_per_layer):
        x_pos = margin + (window_width - 2 * margin) * (
            i / (layer_count - 1 if layer_count > 1 else 1)
        )

        # Calculate the total height occupied by the nodes in this layer
        layer_height = node_spacing * (nodes_in_layer - 1)

        # Calculate the starting y-position to center the nodes vertically
        y_start = margin + (available_height - layer_height) / 2

        for node_idx in range(nodes_in_layer):
            y_pos = y_start + node_idx * node_spacing
            node_positions[(i, node_idx)] = (x_pos, y_pos)

    # Initialize importance dictionary
    node_importance = {
        (i, idx): 0
        for i in range(layer_count)
        for idx in range(num_nodes_per_layer[i])
    }

    # Initialize a set to keep track of nodes that have drawn connections
    drawn_nodes = set()

    # Initialize plot
    fig, ax = plt.subplots(figsize=(window_width, window_height))

    # Draw connections and calculate node importance
    for (i, j), tensor in connections_dict.items():
        # Convert tensor to NumPy array if it's a PyTorch tensor
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        elif not isinstance(tensor, np.ndarray):
            tensor = np.array(tensor)

        # Transpose the tensor to align dimensions
        tensor = tensor.T

        # Iterate through the tensor to find connections to draw
        for start_node_idx in range(num_nodes_per_layer[i]):
            for end_node_idx in range(num_nodes_per_layer[j]):
                weight = tensor[start_node_idx, end_node_idx]
                if abs(weight) > 15:
                    importance = abs(weight)
                    node_importance[(i, start_node_idx)] += importance  # Outgoing
                    node_importance[(j, end_node_idx)] += importance    # Incoming
                    start_pos = node_positions.get((i, start_node_idx))
                    end_pos = node_positions.get((j, end_node_idx))
                    if start_pos and end_pos:
                        x_start, y_start = start_pos
                        x_end, y_end = end_pos
                        # Ensure m_a is defined; replace with your max importance value
                        ax.plot(
                            [x_start, x_end],
                            [y_start, y_end],
                            color=connection_color,
                            alpha=float(importance / m_a),
                        )
                        # Add nodes to the set of drawn nodes
                        drawn_nodes.add((i, start_node_idx))
                        drawn_nodes.add((j, end_node_idx))

    # Get the top N nodes based on importance
    top_nodes = sorted(
        node_importance.items(), key=lambda x: x[1], reverse=True
    )[:top_n]
    top_nodes_set = set([node for node, importance in top_nodes])

    # Prepare feature indices for describe_features function
    features_dict = {}
    for (layer_idx, node_idx), importance in top_nodes:
        if layer_idx not in features_dict:
            features_dict[layer_idx] = []
        features_dict[layer_idx].append(node_idx)

    # Get descriptions using describe_features function
    node_descriptions = {}
    if describe_features_func is not None:
        descriptions = describe_features_func(features_dict)
        for layer_idx, features in descriptions.items():
            for feature in features:
                node_descriptions[(layer_idx, feature['feature_id'])] = feature['description']

    # Draw nodes and labels only for nodes that have drawn connections
    for (i, node_idx), (x, y) in node_positions.items():
        if (i, node_idx) in drawn_nodes:
            ax.scatter(x, y, s=node_size, color=node_color, zorder=5, edgecolors="k")

            # If the node is in top N, add a label
            if (i, node_idx) in top_nodes_set:
                description = node_descriptions.get((i, node_idx), "")
                print(description)
                ax.text(
                    x,
                    y + node_size * 0.002,  # Slight offset above the node
                    f"{description}",
                    fontsize=8,
                    ha='center',
                    va='bottom',
                    zorder=6,  # Ensure text is on top
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor=text_bg_color,
                        edgecolor='none',
                        alpha=text_bg_alpha,
                    ),
                )

    # Set limits and remove axes
    ax.set_xlim(0, window_width)
    ax.set_ylim(0, window_height)
    ax.axis("off")

    plt.tight_layout()
    plt.show()

from feature_circuit_discovery.sae_funcs import (
    describe_features
)

num_nodes = [len(i) for i in activated_features.values()]

plot_layer_connections(connections, num_nodes, describe_features_func = describe_features)
# %%
def plot_tensor_dict_distribution(tensor_dict):
    # Concatenate all tensors' values into a single 1D array
    all_values = torch.cat([v.flatten() for v in tensor_dict.values()]).cpu().numpy()
    
    # Plot histogram
    plt.hist(all_values, bins=300, edgecolor='black', alpha=0.7)
    plt.title("Tensor Dictionary Value Distribution")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.show()
plot_tensor_dict_distribution(connections)

# %%
for i in prompt_data["prompts"][:15]:
    print(i)
# %%

# %%
print(list(connections.values())[0])
# %%
