# %%
import json

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import torch
from feature_circuit_discovery.sae_funcs import (
    compute_gradient_matrix,
    get_active_features,
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
    "/root/feature-circuit-discovery/datasets/ioi/ioi_test_100.json", "rb"
) as file:
    prompt_data = json.load(file)

prompt_data["prompts"] = [
    " ".join(sentence.split()[:-1]) for sentence in prompt_data["sentences"]
]

# %%
with open(
    "/root/feature-circuit-discovery/datasets/ioi/ioi_test_100_2.json", "w"
) as file:
    json.dump(prompt_data, file)


# %%
tokenized_prompts = (
    tokenizer(
        prompt_data["prompts"][:15],
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    )
    .data["input_ids"]
    .to(device)
)

matrices = []

activated_features = get_active_features(tokenized_prompts, model, device, threshold=1)

for i in range(len(activated_features)):
    print()
    print(activated_features[i])
    print("length:",len(activated_features[i]))

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
# %%
for i in matrices:
    print(i)"""

# %%
connections = {}
num_layers = len(activated_features)

for i in tqdm(range(num_layers)):
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
        )
        # Store the computed matrix in the dictionary
        connections[(i, j)] = grad_matrix


# %%


def plot_layer_connections(connections_dict, num_nodes_per_layer, 
                           window_width=10, window_height=10, margin=1, 
                           node_size=50, node_color='white', 
                           connection_color='black', 
                           connection_alpha=0.2):
    
    layer_count = len(num_nodes_per_layer)
    
    # Assign positions to each node
    node_positions = {}
    for i, nodes_in_layer in enumerate(num_nodes_per_layer):
        x_pos = margin + (window_width - 2 * margin) * (i / (layer_count - 1 if layer_count > 1 else 1))
        for node_idx in range(nodes_in_layer):
            y_pos = margin + (window_height - 2 * margin) * (node_idx / (nodes_in_layer - 1 if nodes_in_layer > 1 else 1))
            node_positions[(i, node_idx)] = (x_pos, y_pos)
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(window_width, window_height))
    
    # Draw connections
    for (i, j), tensor in connections_dict.items():
        # Convert tensor to NumPy array if it's a PyTorch tensor
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        elif not isinstance(tensor, np.ndarray):
            tensor = np.array(tensor)
        #print(num_nodes_per_layer[i],num_nodes_per_layer[j], tensor.shape)
        # Iterate through the tensor to find non-zero connections
        for start_node_idx in range(num_nodes_per_layer[i]):
            for end_node_idx in range(num_nodes_per_layer[j]):
                if tensor.T[start_node_idx, end_node_idx] != 0.0:
                    start_pos = node_positions.get((i, start_node_idx))
                    end_pos = node_positions.get((j, end_node_idx))
                    if start_pos and end_pos:
                        x_start, y_start = start_pos
                        x_end, y_end = end_pos
                        ax.plot([x_start, x_end], 
                                [y_start, y_end], 
                                color=connection_color, alpha=connection_alpha)
    
    # Draw nodes
    for (i, node_idx), (x, y) in node_positions.items():
        ax.scatter(x, y, s=node_size, color=node_color, zorder=3, edgecolors='k')
        
    # Set limits and remove axes
    ax.set_xlim(0, window_width)
    ax.set_ylim(0, window_height)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

num_nodes = [len(i) for i in activated_features]

plot_layer_connections(connections, num_nodes)   
# %%

# %%
