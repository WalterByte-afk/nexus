import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

def generate_architecture_diagram():
    """
    Generates a professional diagram showing the 8 layers of NEXUS-Ω.
    """
    layers = [
        "1. Sparse Dynamic Router",
        "2. KAN Activation Edges",
        "3. Predictive Coding",
        "4. Adaptive Recurrent Depth",
        "5. Dual-Plasticity Weights",
        "6. Synaptic Consolidation",
        "7. Engram Memory Cortex",
        "8. Sparse Distributed Input"
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(layers) + 1)

    # Colors for a modern, professional look
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(layers)))

    for i, layer in enumerate(layers):
        # Draw layer block
        rect = plt.Rectangle((2, len(layers) - i), 6, 0.8, color=colors[i], alpha=0.8)
        ax.add_patch(rect)

        # Add text
        ax.text(5, len(layers) - i + 0.4, layer, color='white',
                ha='center', va='center', fontsize=12, fontweight='bold')

        # Draw arrow to next layer
        if i < len(layers) - 1:
            ax.annotate('', xy=(5, len(layers) - i - 0.1),
                        xytext=(5, len(layers) - i),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    ax.set_title("NEXUS-Ω: Hierarchical Layer Architecture", fontsize=16, pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig("architecture_layers.png", dpi=300)
    print("[OK] Created architecture_layers.png")

def generate_neural_network_diagram():
    """
    Generates a graph representing the connectivity and self-updating plasticity.
    """
    G = nx.Graph()

    # Define nodes: base weights (static) and plastic weights (dynamic)
    num_neurons = 15
    nodes = [f"N_{i}" for i in range(num_neurons)]
    G.add_nodes_from(nodes)

    # Create a dense-ish connectivity with some clusters
    for i in range(num_neurons):
        for j in range(i + 1, num_neurons):
            if np.random.rand() < 0.25: # Sparsity
                G.add_edge(nodes[i], nodes[j])

    # Add "Plasticity Feedback" loops
    for i in range(5):
        G.add_edge(nodes[i], nodes[i+1], type='plastic')

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    # Draw edges
    edges = G.edges()
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='gray', alpha=0.5)

    # Highlight plastic edges (the ones that "fire together wire together")
    plastic_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'plastic']
    nx.draw_networkx_edges(G, pos, edgelist=plastic_edges, width=3, edge_color='orange', alpha=0.8)

    plt.title("NEXUS-Ω: Synaptic Plasticity & Neural Connectivity Graph\n(Orange lines = Dynamic Plasticity Updates)", fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("neural_connectivity.png", dpi=300)
    print("[OK] Created neural_connectivity.png")

def generate_memory_diagram():
    """
    Generates a diagram showing the Dual-Plasticity (Hippocampus/Neocortex) memory hierarchy.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Draw two main regions
    # Hippocampus (Fast, Plastic)
    ax.add_patch(plt.Rectangle((1, 1), 3.5, 4, color='salmon', alpha=0.3, label='Hippocampus (Fast)'))
    ax.text(2.75, 4.5, "Hippocampus\n(Plasticity)", ha='center', fontweight='bold')

    # Neocortex (Slow, Stable)
    ax.add_patch(plt.Rectangle((5.5, 1), 3.5, 4, color='skyblue', alpha=0.3, label='Neocortex (Stable)'))
    ax.text(7.25, 4.5, "Neocortex\n(Consolidated)", ha='center', fontweight='bold')

    # Draw "Consolidation" arrow
    ax.annotate('Synaptic Consolidation\n(Sleep Mode)', xy=(5.5, 3), xytext=(4.5, 3),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                ha='center', va='center', fontsize=10)

    # Add details
    ax.text(2.75, 2, "High Plasticity\nShort-term Memory\nFast Updates", ha='center', fontsize=9)
    ax.text(7.25, 2, "Low Plasticity\nLong-term Memory\nStable Weights", ha='center', fontsize=9)

    ax.set_title("NEXUS-Ω: Dual-Plasticity Memory Hierarchy", fontsize=16)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig("memory_hierarchy.png", dpi=300)
    print("[OK] Created memory_hierarchy.png")

if __name__ == "__main__":
    # We need to ensure directories exist
    os.makedirs("docs/diagrams", exist_ok=True)

    # Generate all three
    generate_architecture_diagram()
    generate_neural_network_diagram()
    generate_memory_diagram()

    # Move to correct folder for repo
    import shutil
    shutil.move("architecture_layers.png", "docs/diagrams/architecture_layers.png")
    shutil.move("neural_connectivity.png", "docs/diagrams/neural_connectivity.png")
    shutil.move("memory_hierarchy.png", "docs/diagrams/memory_hierarchy.png")

    print("\n[DONE] All diagrams generated and saved to docs/diagrams/")
