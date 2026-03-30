"""
Visualization tools for feature embeddings.

Provides three complementary views:
1. t-SNE 2D projection (reveals clustering structure)
2. PCA 2D projection (reveals main variance directions)
3. Pairwise cosine similarity heatmap (reveals coupling strength)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine, pdist, squareform


class EmbeddingVisualizer:
    """Visualize feature embeddings in multiple ways."""
    
    @staticmethod
    def plot_tsne(embeddings: np.ndarray, feature_names: list,
                  perplexity: int = 5, save_path: str = None):
        """
        t-SNE projection to 2D.
        
        Best for: Revealing clusters of physically related sensors.
        
        :param embeddings: [N, d_model] embedding matrix.
        :param feature_names: List of N sensor names.
        :param perplexity: t-SNE perplexity (lower for small datasets).
        :param save_path: Where to save the figure.
        """
        n_features = len(embeddings)
        
        # Adjust perplexity if dataset is too small
        perplexity = min(perplexity, n_features - 1)
        
        print(f"🔄 Running t-SNE (perplexity={perplexity})...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords = tsne.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Color by sensor type if naming convention allows
        # Example: mass_1_pos, mass_1_vel, mass_1_acc
        colors = []
        for name in feature_names:
            if 'pos' in name.lower():
                colors.append('#1f77b4')  # Blue
            elif 'vel' in name.lower():
                colors.append('#ff7f0e')  # Orange
            elif 'acc' in name.lower():
                colors.append('#2ca02c')  # Green
            else:
                colors.append('#d62728')  # Red
        
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=100, alpha=0.7, edgecolors='k')
        
        for i, name in enumerate(feature_names):
            ax.annotate(name, (coords[i, 0], coords[i, 1]),
                       fontsize=8, ha='center', va='bottom')
        
        ax.set_title("Feature Embedding Space (t-SNE Projection)", fontsize=16, weight='bold')
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', label='Position'),
            Patch(facecolor='#ff7f0e', label='Velocity'),
            Patch(facecolor='#2ca02c', label='Acceleration'),
            Patch(facecolor='#d62728', label='Other')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✅ t-SNE plot saved: {save_path}")
        return fig
    
    @staticmethod
    def plot_pca(embeddings: np.ndarray, feature_names: list,
                 save_path: str = None):
        """
        PCA projection to 2D.
        
        Best for: Understanding which features vary together.
        
        :param embeddings: [N, d_model] embedding matrix.
        :param feature_names: List of N sensor names.
        :param save_path: Where to save the figure.
        """
        print("🔄 Running PCA...")
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Same coloring scheme as t-SNE
        colors = []
        for name in feature_names:
            if 'pos' in name.lower():
                colors.append('#1f77b4')
            elif 'vel' in name.lower():
                colors.append('#ff7f0e')
            elif 'acc' in name.lower():
                colors.append('#2ca02c')
            else:
                colors.append('#d62728')
        
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=100, alpha=0.7, edgecolors='k')
        
        for i, name in enumerate(feature_names):
            ax.annotate(name, (coords[i, 0], coords[i, 1]),
                       fontsize=8, ha='center', va='bottom')
        
        var_explained = pca.explained_variance_ratio_
        ax.set_title(
            f"Feature Embedding Space (PCA Projection)\n"
            f"PC1: {var_explained[0]:.1%} var, PC2: {var_explained[1]:.1%} var",
            fontsize=16, weight='bold'
        )
        ax.set_xlabel(f"Principal Component 1 ({var_explained[0]:.1%})")
        ax.set_ylabel(f"Principal Component 2 ({var_explained[1]:.1%})")
        ax.grid(True, alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', label='Position'),
            Patch(facecolor='#ff7f0e', label='Velocity'),
            Patch(facecolor='#2ca02c', label='Acceleration'),
            Patch(facecolor='#d62728', label='Other')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✅ PCA plot saved: {save_path}")
        return fig
    
    @staticmethod
    def plot_similarity_heatmap(embeddings: np.ndarray, feature_names: list,
                                 save_path: str = None):
        """
        Cosine similarity heatmap between all feature pairs.
        
        Best for: Identifying which sensors are "coupled" in the learned space.
        
        :param embeddings: [N, d_model] embedding matrix.
        :param feature_names: List of N sensor names.
        :param save_path: Where to save the figure.
        """
        print("🔄 Computing pairwise cosine similarities...")
        
        # Compute cosine similarity matrix
        n = len(embeddings)
        similarity = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity[i, j] = 1.0
                else:
                    similarity[i, j] = 1 - cosine(embeddings[i], embeddings[j])
        
        fig, ax = plt.subplots(figsize=(18, 16))
        sns.heatmap(similarity, xticklabels=feature_names, yticklabels=feature_names,
                   cmap='coolwarm', center=0, square=True, ax=ax,
                   cbar_kws={'label': 'Cosine Similarity'})
        
        ax.set_title("Feature Embedding Similarity Matrix", fontsize=18, weight='bold')
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✅ Similarity heatmap saved: {save_path}")
        return fig
    
    @staticmethod
    def plot_all(embeddings: np.ndarray, feature_names: list,
                 output_dir: str = "."):
        """
        Generate all three visualizations at once.
        
        :param embeddings: [N, d_model] embedding matrix.
        :param feature_names: List of N sensor names.
        :param output_dir: Directory to save figures.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        EmbeddingVisualizer.plot_tsne(
            embeddings, feature_names,
            save_path=os.path.join(output_dir, "embedding_tsne.png")
        )
        plt.close()
        
        EmbeddingVisualizer.plot_pca(
            embeddings, feature_names,
            save_path=os.path.join(output_dir, "embedding_pca.png")
        )
        plt.close()
        
        EmbeddingVisualizer.plot_similarity_heatmap(
            embeddings, feature_names,
            save_path=os.path.join(output_dir, "embedding_similarity.png")
        )
        plt.close()
        
        print(f"\n✅ All embedding visualizations saved to: {output_dir}")


def find_most_similar(embeddings: np.ndarray, feature_names: list,
                      query_idx: int, top_k: int = 5) -> list:
    """
    Find the top-k most similar features to a query feature.
    
    Useful for: Identifying which sensors are coupled to a specific sensor.
    
    :param embeddings: [N, d_model] embedding matrix.
    :param feature_names: List of N sensor names.
    :param query_idx: Index of the query feature.
    :param top_k: Number of similar features to return.
    :returns: List of (name, similarity_score) tuples.
    """
    query_emb = embeddings[query_idx]
    
    similarities = []
    for i in range(len(embeddings)):
        if i == query_idx:
            continue
        sim = 1 - cosine(query_emb, embeddings[i])
        similarities.append((feature_names[i], sim, i))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
