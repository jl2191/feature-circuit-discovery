"""Generate a feature circuit graph for 3-digit addition prompts."""
import random
import gc
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
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
print(f"Prompts: {prompts[:3]}...")

inputs = tokenizer.encode(prompts[0], return_tensors="pt", add_special_tokens=True).to(device)

print("\nGetting active features...")
activated_features = get_active_features(prompts, tokenizer, model, device)

# Use first 30 features per layer for layers 0→1 and 1→2
N_FEATS = 30
layer_pairs = [(0, 1), (1, 2)]

for up_layer, down_layer in layer_pairs:
    up_feats = activated_features[up_layer][:N_FEATS]
    down_feats = activated_features[down_layer][:N_FEATS]
    print(f"\n--- Layer {up_layer} → {down_layer} ---")
    print(f"  Upstream: {len(up_feats)} features, Downstream: {len(down_feats)} features")

    grad_matrix = compute_gradient_matrix(
        inputs, up_layer, down_layer, up_feats, down_feats, model, verbose=True,
    )

    print(f"  max={grad_matrix.abs().max():.4f}, mean={grad_matrix.abs().mean():.4f}, "
          f"nonzero={grad_matrix.count_nonzero()}/{grad_matrix.numel()}")

    # --- Draw graph ---
    tensor = grad_matrix.cpu()
    threshold = 0.01  # lower threshold to show more edges
    n_down, n_up = tensor.shape

    B = nx.Graph()
    left_nodes = [f"L{up_layer}_f{up_feats[i].item()}" for i in range(n_up)]
    right_nodes = [f"L{down_layer}_f{down_feats[j].item()}" for j in range(n_down)]
    B.add_nodes_from(left_nodes, bipartite=0)
    B.add_nodes_from(right_nodes, bipartite=1)

    for j in range(n_down):
        for i in range(n_up):
            w = tensor[j, i].item()
            if abs(w) >= threshold:
                B.add_edge(left_nodes[i], right_nodes[j], weight=abs(w))

    pos = nx.bipartite_layout(B, left_nodes)
    edges = B.edges(data=True)
    edge_weights = np.array([e[2]["weight"] for e in edges])

    if len(edge_weights) == 0:
        print(f"  No edges above threshold={threshold}")
        continue

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle(f"Feature Circuit: Layer {up_layer} → Layer {down_layer}\n"
                 f"(3-digit addition, {len(edge_weights)} edges, threshold={threshold})",
                 fontsize=14, fontweight="bold")

    norm = plt.Normalize(vmin=edge_weights.min(), vmax=edge_weights.max())
    colors = cm.viridis(norm(edge_weights))

    # Draw nodes
    nx.draw_networkx_nodes(B, pos, nodelist=left_nodes, node_color="lightcoral",
                           node_size=300, ax=ax, label=f"Layer {up_layer}")
    nx.draw_networkx_nodes(B, pos, nodelist=right_nodes, node_color="lightblue",
                           node_size=300, ax=ax, label=f"Layer {down_layer}")

    # Draw edges with width proportional to gradient magnitude
    max_w = edge_weights.max()
    for (u, v, d), color in zip(B.edges(data=True), colors):
        width = 0.5 + 4.0 * (d["weight"] / max_w)
        nx.draw_networkx_edges(B, pos, edgelist=[(u, v)], width=width,
                               edge_color=[color], alpha=0.7, ax=ax)

    # Compact labels (just feature index)
    labels = {}
    for n in left_nodes:
        labels[n] = n.split("_f")[1]
    for n in right_nodes:
        labels[n] = n.split("_f")[1]
    nx.draw_networkx_labels(B, pos, labels, font_size=6, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label="|Gradient|", ax=ax, shrink=0.8)
    ax.legend(loc="upper left", fontsize=10)

    outpath = f"experiments/circuit_L{up_layer}_L{down_layer}.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")

    del grad_matrix
    gc.collect()

print("\nDone!")
