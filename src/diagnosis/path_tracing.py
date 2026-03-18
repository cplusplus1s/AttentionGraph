"""
Attention path tracing for root cause analysis.

Provides two complementary tracing strategies:
1. Backward tracing: From anomalous sensor (Victim) → find root cause
2. Forward BFS tracing: From suspected root (Epicenter) → see propagation tree
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional

from .base import BaseDiagnoser, DiagnosisResult


class PathTracingDiagnoser(BaseDiagnoser):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.baseline_map: Optional[np.ndarray] = None
        self.threshold = config.get('path_threshold', 0.05)
        self.max_depth = config.get('max_depth', 5)

        self.top_k_starts = config.get('top_k_starts', 3)  # number of backward start points
        self.max_bfs_branches = config.get('max_bfs_branches', 3)  # max number of forward branches per level

    def fit(self, normal_attention_maps: np.ndarray):
        print(f"🧠 [PathTracer] Learning baseline from {len(normal_attention_maps)} samples...")
        self.baseline_map = np.mean(normal_attention_maps, axis=0)

    def diagnose(self, current_attention_map: np.ndarray,
                 root_candidates: List[int] = None,
                 prediction_error: Optional[np.ndarray] = None) -> DiagnosisResult:
        if self.baseline_map is None:
            raise ValueError("PathTracer not fitted! Call fit() first.")

        diff_matrix = np.abs(current_attention_map - self.baseline_map)
        all_paths = []

        # 1. Backward Tracing (start with the victims, trace the root cause)
        if prediction_error is not None:
            start_sensors = np.argsort(prediction_error)[-self.top_k_starts:][::-1]
        else:
            row_drifts = np.sum(diff_matrix, axis=1)
            start_sensors = np.argsort(row_drifts)[-self.top_k_starts:][::-1]

        for start_idx in start_sensors:
            path_info = self._trace_backward(current_attention_map, start_idx, depth=self.max_depth)
            if len(path_info['nodes']) > 1:
                all_paths.append(path_info)

        # 2. Forward BFS Tracing (start from SpectralAttentionDriftDiagnoser root_candidates)
        if root_candidates:
            for root_idx in root_candidates:
                f_paths = self._trace_forward_bfs(diff_matrix, root_idx, depth=self.max_depth)
                all_paths.extend(f_paths)

        # 3. Build diagnosis result
        is_anomaly = len(all_paths) > 0
        severity = self._compute_path_severity(all_paths, diff_matrix)
        description = self._format_path_description(all_paths)

        evidence = []
        for path_info in all_paths:
            evidence.append({
                'type': 'propagation_path',
                'trace_type': path_info['trace_type'],
                'path': [self.feature_names[i] for i in path_info['nodes']],
                'path_strength': path_info['strengths'],
                'root_cause_candidate': self.feature_names[path_info['nodes'][0]] # Index 0 is always root now
            })

        return DiagnosisResult(
            is_anomaly=is_anomaly,
            severity=severity,
            diagnosis_type="PathTracing",
            description=description,
            evidence=evidence,
            details={'all_paths': all_paths}
        )

    def _trace_backward(self, attention_map: np.ndarray, start_idx: int, depth: int) -> Dict:
        """Backward tracing, start from victims, extract the strongest contributors"""
        path = [start_idx]
        strengths = []
        current_idx = start_idx

        for _ in range(depth):
            sources = attention_map[current_idx, :].copy()
            sources[current_idx] = 0
            for visited in path:
                sources[visited] = 0

            strongest_source = np.argmax(sources)
            strength = sources[strongest_source]

            if strength < self.threshold:
                break

            path.append(strongest_source)
            strengths.append(float(strength))
            current_idx = strongest_source

        # Reverse victim -> root_cause chain to root_cause -> victim chain
        return {
            'nodes': path[::-1],
            'strengths': strengths[::-1],
            'trace_type': 'Backward (Victim)'
        }

    def _trace_forward_bfs(self, diff_matrix: np.ndarray, root_idx: int, depth: int) -> List[Dict]:
        """Forward tracing, start from Spectral root_candidates, breadth-first search"""
        paths = []
        queue = [([root_idx], [])]

        while queue:
            current_path, current_strengths = queue.pop(0)

            if len(current_path) - 1 >= depth:
                if len(current_path) > 1:
                    paths.append({'nodes': current_path, 'strengths': current_strengths, 'trace_type': 'Forward (Spectral Root)'})
                continue

            curr_node = current_path[-1]

            # Search for drift > threshold
            neighbors = []
            for v in range(diff_matrix.shape[1]):
                if v not in current_path and diff_matrix[curr_node, v] >= self.threshold:
                    neighbors.append(v)

            # Path reaches end
            if not neighbors:
                if len(current_path) > 1:
                    paths.append({'nodes': current_path, 'strengths': current_strengths, 'trace_type': 'Forward (Spectral Root)'})
            else:
                # Choose top-k impacted neighbors
                neighbors = sorted(neighbors, key=lambda x: diff_matrix[curr_node, x], reverse=True)[:self.max_bfs_branches]
                for v in neighbors:
                    queue.append((current_path + [v], current_strengths + [float(diff_matrix[curr_node, v])]))

        return paths

    def _compute_path_severity(self, paths: List[Dict], diff_matrix: np.ndarray) -> float:
        if not paths: return 0.0
        max_severity = 0.0
        for path_info in paths:
            nodes = path_info['nodes']
            if len(nodes) < 2: continue
            path_drift = sum(diff_matrix[nodes[i], nodes[i+1]] for i in range(len(nodes) - 1))
            max_severity = max(max_severity, path_drift / (len(nodes) - 1))
        return min(max_severity * 2, 1.0)

    def _format_path_description(self, paths: List[Dict]) -> str:
        if not paths: return "No significant attention paths detected."
        desc_lines = [f"Detected {len(paths)} anomalous propagation path(s):"]
        for i, path_info in enumerate(paths, 1):
            nodes = path_info['nodes']
            if len(nodes) < 2: continue
            path_str = " → ".join([self.feature_names[idx] if self.feature_names else f"F{idx}" for idx in nodes])
            desc_lines.append(f"  [{path_info['trace_type']}] Path {i}: {path_str}")
        return "\n".join(desc_lines)

    def visualize_paths(self, current_attention_map: np.ndarray, diagnosis_result: DiagnosisResult, save_dir: str = None):
        import matplotlib.pyplot as plt
        import networkx as nx
        import os

        if not save_dir:
            return

        os.makedirs(save_dir, exist_ok=True)
        all_paths = diagnosis_result.details.get('all_paths', [])

        def _draw_graph(paths_to_draw, title, filename, default_color):
            G = nx.DiGraph()
            edge_weights = {}
            for p in paths_to_draw:
                nodes = p['nodes']
                strengths = p['strengths']
                for i in range(len(nodes) - 1):
                    source = nodes[i]
                    target = nodes[i+1]
                    weight = strengths[i] if i < len(strengths) else 0

                    edge_key = (source, target)
                    edge_weights[edge_key] = max(edge_weights.get(edge_key, 0), weight)

            if not edge_weights:
                return

            for (u, v), w in edge_weights.items():
                G.add_edge(u, v, weight=w)

            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(G, k=2.0, seed=42)
            labels = {i: self.feature_names[i] if self.feature_names else f"F{i}" for i in G.nodes()}

            nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='#E8F0FE', edgecolors='#1A73E8', linewidths=2.5, ax=ax)
            nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold', ax=ax)

            weights = [G[u][v]['weight'] for u, v in G.edges()]
            if weights:
                max_w = max(weights) if max(weights) > 0 else 1
                widths = [2 + 6 * (w / max_w) for w in weights]

                nx.draw_networkx_edges(
                    G, pos, width=widths, edge_color=default_color,
                    arrowsize=35, arrowstyle='-|>',
                    connectionstyle='arc3,rad=0.15', ax=ax
                )

            ax.set_title(title, fontsize=16, weight='bold', pad=15)
            ax.axis('off')
            plt.tight_layout()

            full_path = os.path.join(save_dir, filename)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"   ✅ Saved: {filename}")

        print("🎨 Generating comprehensive path visualizations...")

        backward_paths = [p for p in all_paths if 'Backward' in p['trace_type']]
        if backward_paths:
            _draw_graph(backward_paths, "Fault Propagation Tree (Backward / Victim)", "tree_01_backward.png", "blue")

        forward_paths = [p for p in all_paths if 'Forward' in p['trace_type']]
        if forward_paths:
            _draw_graph(forward_paths, "Fault Propagation Tree (Forward / Spectral Root)", "tree_02_forward.png", "red")

        for i, p in enumerate(all_paths, 1):
            trace_type = "Forward" if "Forward" in p['trace_type'] else "Backward"
            color = "red" if trace_type == "Forward" else "blue"
            title = f"Path {i} ({trace_type})"
            filename = f"path_{i:02d}_{trace_type.lower()}.png"
            _draw_graph([p], title, filename, color)