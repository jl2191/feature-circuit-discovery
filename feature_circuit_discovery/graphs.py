import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm


def draw_bipartite_graph(tensor, threshold=0.05):
    tensor = tensor.cpu()
    # Set your threshold
    threshold = 0.05

    m, n = tensor.shape

    # Create a bipartite graph
    B = nx.Graph()

    # Add m nodes on one side (left)
    left_nodes = [f"left_{i}" for i in range(m)]
    B.add_nodes_from(left_nodes, bipartite=0)

    # Add n nodes on the other side (right)
    right_nodes = [f"right_{j}" for j in range(n)]
    B.add_nodes_from(right_nodes, bipartite=1)

    # Add edges with weights from the tensor (skip edges below threshold)
    for i in range(m):
        for j in range(n):
            if (
                abs(tensor[i, j]) >= threshold
            ):  # Only add edges with weights >= threshold
                B.add_edge(left_nodes[i], right_nodes[j], weight=tensor[i, j])

    # Draw the graph
    pos = nx.bipartite_layout(B, left_nodes)
    edges = B.edges(data=True)
    edge_weights = np.array([edge[2]["weight"] for edge in edges])

    # Normalize weights to the range [0, 1] for coloring
    norm = plt.Normalize(vmin=edge_weights.min(), vmax=edge_weights.max())
    colors = cm.viridis(norm(edge_weights))

    # Draw the nodes
    nx.draw(B, pos, with_labels=True, node_color="lightblue", node_size=500)

    # Draw the edges with transparency (alpha) and colors based on strength
    for (u, v, d), color in zip(edges, colors):
        nx.draw_networkx_edges(
            B,
            pos,
            edgelist=[(u, v)],
            width=d["weight"] * 5,
            edge_color=[color],
            alpha=0.8,
        )

    # Display the plot
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label="Connection Strength")
    plt.show()
