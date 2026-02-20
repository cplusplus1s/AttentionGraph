import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np

class Visualizer:
    def plot_heatmap(self, matrix, labels, title="Attention Heatmap"):
        plt.figure(figsize=(20, 18))
        sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap='viridis', square=True)
        plt.title(title, fontsize=20)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        return plt.gcf()

    def plot_graph(self, G, title="Topology Graph", layout_type='spring'):
        plt.figure(figsize=(16, 14))

        if layout_type == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, k=0.5, seed=42)

        d = dict(G.degree(weight='weight'))
        node_sizes = [min(np.sqrt(d.get(n, 0)) * 800 + 500, 3000) for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

        # Adjust the edge thickness according to the weight
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        if weights:
            max_w, min_w = max(weights), min(weights)
            widths = [1 + 4 * (w - min_w) / (max_w - min_w + 1e-9) for w in weights]
            nx.draw_networkx_edges(G, pos, width=widths, edge_color='gray', arrowsize=20, alpha=0.6)

        plt.title(title, fontsize=18)
        plt.axis('off')
        plt.tight_layout()
        return plt.gcf()