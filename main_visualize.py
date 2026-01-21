import yaml
import json
import os
import sys
import src

from src.analysis.result_loader import ResultLoader
from src.analysis.graph_builder import GraphBuilder
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

    # 2. Load data
    loader = ResultLoader(
        results_root=config['paths']['results_dir'],
        data_path=config['paths']['processed_csv']
    )
    matrix, sensor_names = loader.load_data()

    # Figures saved to: ./results/sensor_analysis_xxx/figures/
    experiment_dir = loader.latest_folder
    output_dir = os.path.join(experiment_dir, "figures")
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

    # 4. Plot and save
    viz = Visualizer()

    # figure 1: heatmap
    fig1 = viz.plot_heatmap(matrix, sensor_names)
    fig1.savefig(os.path.join(output_dir, "heatmap.png"), dpi=300)
    plt.close(fig1)
    print("✅ Heatmap saved.")

    # figure 2: signal topology
    fig2 = viz.plot_graph(G_signal, title="Signal Topology")
    fig2.savefig(os.path.join(output_dir, "signal_graph.png"), dpi=300)
    plt.close(fig2)
    print("✅ Signal Graph saved.")

    # figure 3: module topology
    fig3 = viz.plot_graph(G_module, title="System Module Topology", layout_type='circular')
    fig3.savefig(os.path.join(output_dir, "module_graph.png"), dpi=300)
    plt.close(fig3)
    print("✅ Module Graph saved.")

if __name__ == "__main__":
    main()