import yaml
import json
import os
import sys
import numpy as np
import src

from src.analysis.result_loader import load_sensor_names, load_attention_weights
from src.analysis.graph_builder import GraphBuilder
from src.analysis.data_exporter import DataExporter
from src.visualization.plotters import Visualizer
from src.visualization.styles import set_style
import matplotlib.pyplot as plt

def main():
    set_style()

    # 1. Load settings
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open('config/sensor_mapping.json', 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    # Batch results
    data_path = config['paths']['processed_csv']
    if not os.path.exists(data_path):
        base, ext = os.path.splitext(data_path)
        data_path_fallback = f"{base}_1{ext}"
        if os.path.exists(data_path_fallback):
            data_path = data_path_fallback
        else:
            raise FileNotFoundError(f"Neither {config['paths']['processed_csv']} nor {data_path_fallback} exists.")

    # 2. Load data — analyze the healthy baseline (aggregated across all runs)
    target_folder = os.path.join(config['paths']['results_dir'], 'healthy_baseline')
    print(f"📂 Analyzing result folder: {target_folder}")

    sensor_names = load_sensor_names(data_path)
    n_sensors = len(sensor_names)
    all_samples = load_attention_weights(target_folder, n_sensors)
    matrix = np.mean(all_samples, axis=0)

    # Figures saved to: ./results/healthy_baseline/figures/
    output_dir = os.path.join(target_folder, "figures")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 The figures will be saved to: {output_dir}")

    # 3. Build graph
    builder = GraphBuilder()
    G_signal = builder.build_signal_graph(
        matrix, sensor_names,
        threshold_std=config['analysis']['threshold_std']
    )
    G_module = builder.build_module_graph(
        matrix, sensor_names, mapping,
        threshold_offset=config['analysis']['module_threshold_offset']
    )

    # 4. Export CSV Data
    exporter = DataExporter()
    exporter.export_graph_to_csv(
        G=G_signal,
        output_path=os.path.join(output_dir, "signal_topology.csv"),
        source_name="Source_Sensor",
        target_name="Target_Sensor",
        weight_name="Attention_Weight"
    )
    exporter.export_graph_to_csv(
        G=G_module,
        output_path=os.path.join(output_dir, "module_topology.csv"),
        source_name="Source_Module",
        target_name="Target_Module",
        weight_name="Aggregated_Weight"
    )

    # 5. Plot and save images
    viz = Visualizer()

    fig1 = viz.plot_heatmap(matrix, sensor_names)
    fig1.savefig(os.path.join(output_dir, "heatmap.png"), dpi=300)
    plt.close(fig1)
    print("✅ Heatmap saved.")

    fig2 = viz.plot_graph(G_signal, title="Signal Topology")
    fig2.savefig(os.path.join(output_dir, "signal_graph.png"), dpi=300)
    plt.close(fig2)
    print("✅ Signal Graph saved.")

    fig3 = viz.plot_graph(G_module, title="System Module Topology", layout_type='circular')
    fig3.savefig(os.path.join(output_dir, "module_graph.png"), dpi=300)
    plt.close(fig3)
    print("✅ Module Graph saved.")

if __name__ == "__main__":
    main()