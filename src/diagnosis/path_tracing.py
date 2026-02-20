"""
Attention path tracing for root cause analysis.

Provides two complementary tracing strategies:
1. Backward tracing: From anomalous sensor → find root cause
2. Forward tracing: From suspected root → see propagation path
"""

import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional

from .base import BaseDiagnoser, DiagnosisResult


class PathTracingDiagnoser(BaseDiagnoser):
    """
    Trace attention paths to identify fault propagation chains.
    
    Two modes:
    - Backward: Start from an anomalous sensor, trace strongest influences back
    - Forward: Start from a suspected root cause, see forward propagation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.baseline_map: Optional[np.ndarray] = None
        self.threshold = config.get('path_threshold', 0.05)
        self.max_depth = config.get('max_depth', 5)
    
    def fit(self, normal_attention_maps: np.ndarray):
        """Learn baseline attention map from normal data."""
        print(f"🧠 [PathTracer] Learning baseline from {len(normal_attention_maps)} samples...")
        self.baseline_map = np.mean(normal_attention_maps, axis=0)
    
    def diagnose(self, current_attention_map: np.ndarray,
                 prediction_error: Optional[np.ndarray] = None) -> DiagnosisResult:
        """
        Trace paths to find root causes.
        
        If prediction_error is provided, starts from the worst-performing sensor.
        Otherwise, uses attention drift to find starting points.
        """
        if self.baseline_map is None:
            raise ValueError("PathTracer not fitted! Call fit() first.")
        
        # 1. Find anomalous sensors (starting points for backward trace)
        diff_matrix = np.abs(current_attention_map - self.baseline_map)
        
        if prediction_error is not None:
            # Start from sensor with highest prediction error
            start_sensors = np.argsort(prediction_error)[-3:][::-1]  # Top 3
        else:
            # Start from sensors with highest attention drift
            row_drifts = np.sum(diff_matrix, axis=1)
            start_sensors = np.argsort(row_drifts)[-3:][::-1]
        
        # 2. Trace paths backward for each anomalous sensor
        all_paths = []
        for start_idx in start_sensors:
            path = self._trace_backward(
                current_attention_map,
                start_idx,
                depth=self.max_depth
            )
            if len(path['nodes']) > 1:  # Non-trivial path
                all_paths.append(path)
        
        # 3. Build diagnosis result
        is_anomaly = len(all_paths) > 0
        severity = self._compute_path_severity(all_paths, diff_matrix)
        
        description = self._format_path_description(all_paths)
        
        # Evidence: Include all traced paths
        evidence = []
        for path_info in all_paths:
            evidence.append({
                'type': 'propagation_path',
                'path': [self.feature_names[i] for i in path_info['nodes']],
                'path_strength': path_info['strengths'],
                'root_cause_candidate': self.feature_names[path_info['nodes'][-1]]
            })
        
        return DiagnosisResult(
            is_anomaly=is_anomaly,
            severity=severity,
            diagnosis_type="PathTracing",
            description=description,
            evidence=evidence,
            details={'all_paths': all_paths}
        )
    
    def _trace_backward(self, attention_map: np.ndarray,
                        start_idx: int, depth: int) -> Dict:
        """
        Trace backward through strongest attention sources.
        
        Returns a path from the anomalous sensor back to its root cause.
        """
        path = [start_idx]
        strengths = []
        current_idx = start_idx
        
        for _ in range(depth):
            # Find strongest source influencing current sensor
            sources = attention_map[current_idx, :].copy()  # Incoming attention
            sources[current_idx] = 0  # Exclude self-attention
            
            # Also zero out already-visited nodes to avoid cycles
            for visited in path:
                sources[visited] = 0
            
            strongest_source = np.argmax(sources)
            strength = sources[strongest_source]
            
            if strength < self.threshold:
                break  # No strong contributor
            
            path.append(strongest_source)
            strengths.append(float(strength))
            current_idx = strongest_source
        
        return {
            'nodes': path,  # [anomaly, ..., root_cause]
            'strengths': strengths
        }
    
    def trace_forward(self, attention_map: np.ndarray,
                      root_idx: int, depth: int = 3) -> Dict:
        """
        Trace forward from a suspected root cause to see propagation.
        
        Public method for exploratory analysis.
        
        :param attention_map: Current attention matrix [N, N].
        :param root_idx: Index of suspected root cause sensor.
        :param depth: How many hops to trace forward.
        :returns: Dict with 'nodes' (path) and 'strengths'.
        """
        path = [root_idx]
        strengths = []
        current_idx = root_idx
        
        for _ in range(depth):
            # Find strongest target influenced by current sensor
            targets = attention_map[:, current_idx].copy()  # Outgoing attention
            targets[current_idx] = 0
            
            for visited in path:
                targets[visited] = 0
            
            strongest_target = np.argmax(targets)
            strength = targets[strongest_target]
            
            if strength < self.threshold:
                break
            
            path.append(strongest_target)
            strengths.append(float(strength))
            current_idx = strongest_target
        
        return {
            'nodes': path,
            'strengths': strengths
        }
    
    def _compute_path_severity(self, paths: List[Dict],
                                diff_matrix: np.ndarray) -> float:
        """Compute overall severity based on traced paths."""
        if not paths:
            return 0.0
        
        # Severity = mean attention change along the strongest path
        max_severity = 0.0
        for path_info in paths:
            nodes = path_info['nodes']
            if len(nodes) < 2:
                continue
            
            # Sum drift along path
            path_drift = 0.0
            for i in range(len(nodes) - 1):
                path_drift += diff_matrix[nodes[i], nodes[i+1]]
            
            path_severity = path_drift / (len(nodes) - 1)
            max_severity = max(max_severity, path_severity)
        
        return min(max_severity * 2, 1.0)  # Normalize to [0, 1]
    
    def _format_path_description(self, paths: List[Dict]) -> str:
        """Format traced paths into a readable description."""
        if not paths:
            return "No significant attention paths detected."
        
        desc_lines = [f"Detected {len(paths)} anomalous propagation path(s):"]
        
        for i, path_info in enumerate(paths, 1):
            nodes = path_info['nodes']
            if len(nodes) < 2:
                continue
            
            path_str = " → ".join([
                self.feature_names[idx] if self.feature_names else f"F{idx}"
                for idx in reversed(nodes)  # Show root → anomaly
            ])
            
            desc_lines.append(f"  Path {i}: {path_str}")
        
        return "\n".join(desc_lines)
    
    def visualize_paths(self, current_attention_map: np.ndarray,
                        diagnosis_result: DiagnosisResult,
                        save_path: str = None):
        """
        Visualize traced paths as a directed graph.
        
        :param current_attention_map: The attention matrix used for diagnosis.
        :param diagnosis_result: Output from self.diagnose().
        :param save_path: Where to save the figure.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        G = nx.DiGraph()
        
        # Extract all paths from evidence
        all_paths = diagnosis_result.details.get('all_paths', [])
        
        # Build graph from paths
        edge_weights = {}
        for path_info in all_paths:
            nodes = path_info['nodes']
            strengths = path_info['strengths']
            
            for i in range(len(nodes) - 1):
                source = nodes[i+1]  # Reversed: root → anomaly
                target = nodes[i]
                weight = strengths[i] if i < len(strengths) else 0
                
                edge_key = (source, target)
                edge_weights[edge_key] = max(edge_weights.get(edge_key, 0), weight)
        
        # Add edges
        for (u, v), w in edge_weights.items():
            G.add_edge(u, v, weight=w)
        
        # Draw
        fig, ax = plt.subplots(figsize=(16, 12))
        pos = nx.spring_layout(G, k=1.5, seed=42)
        
        # Node labels
        labels = {i: self.feature_names[i] if self.feature_names else f"F{i}"
                 for i in G.nodes()}
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue',
                              edgecolors='black', linewidths=2, ax=ax)
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax)
        
        # Draw edges with varying thickness
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        if weights:
            max_w = max(weights)
            widths = [1 + 5 * (w / max_w) for w in weights]
            nx.draw_networkx_edges(G, pos, width=widths, edge_color='red',
                                  arrowsize=20, arrowstyle='->', ax=ax)
        
        ax.set_title("Fault Propagation Paths (Path Tracing)", fontsize=16, weight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✅ Path visualization saved: {save_path}")
        
        return fig
