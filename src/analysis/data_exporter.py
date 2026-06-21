import os
import pandas as pd
import networkx as nx

class DataExporter:
    """
    Export attention graph to text file.
    """

    @staticmethod
    def export_graph_to_csv(G: nx.DiGraph, output_path: str,
                            source_name: str = "Source",
                            target_name: str = "Target",
                            weight_name: str = "Weight") -> None:
        edge_data = []

        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 0.0)
            edge_data.append({
                source_name: u,
                target_name: v,
                weight_name: weight
            })

        df = pd.DataFrame(edge_data)

        if df.empty:
            df = pd.DataFrame(columns=[source_name, target_name, weight_name])
        else:
            df = df.sort_values(by=weight_name, ascending=False)

        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ Data exported: {os.path.basename(output_path)}")