# AttentionGraph

## Overview
This project aims to perform data analysis and fault diagnosis of engineering systems.

## 📂 Project Structure

```text
AttentionGraph/
├── config/                 # Configuration files
│   ├── settings.yaml       # Global settings (paths, thresholds, preprocessing)
│   └── sensor_mapping.json # Mapping sensors to system modules (e.g., Gas, RF)
├── data/
│   ├── raw/                # Original WDL files
│   └── processed/          # Aligned and resampled CSVs
├── results/                # Experiment outputs (Model weights, Logs, Figures)
├── run_scripts/            # Execution scripts (PowerShell/Shell)
│   └── run_etch.ps1        # Script to trigger iTransformer training
├── src/                    # Source code
│   ├── analysis/           # Graph construction & result loading logic
│   ├── etl/                # Data preprocessing (ETL) logic
│   └── visualization/      # Plotting utilities
├── third_party/            # External submodules
│   └── iTransformer/       # Modified iTransformer source code
├── main_pipeline.py        # Entry point for Data Preprocessing
├── main_visualize.py       # Entry point for Graph Visualization
└── requirements.txt        # Python dependencies
```

## 🛠️ Environment Setup

### Requirements
- **Python**: 3.10+
- **CUDA**: 11.8+ (Recommended for GPU acceleration)

### Installation
The environment configuration is specified in `requirements.txt`.

1. **Create a virtual environment (Conda recommended):**
   ```bash
   conda create -n attention_graph python=3.10 -y
   conda activate attention_graph
   ```
2. **Install dependencies:**
   ```bash
   # Install PyTorch with CUDA support (Adjust version based on your GPU)
   pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
   # Install project requirements
   pip install -r requirements.txt
   ```

## 🚀 Usage

Follow these steps to run the complete pipeline from raw data to visualization.

### Step 1: Data Preprocessing
Clean, align, and resample the raw data.
```bash
python main_pipeline.py
```
- Configuration: Controlled by config/settings.yaml.
- Process: Reads raw data, performs time alignment, resamples (e.g., to 200ms), and selects specific sensors.
- Output: Generates data/processed/custom_aligned.csv.

### Step 2: Model Training
Train the iTransformer model to capture temporal dependencies and extract attention weights.
```PowerShell
.\run_scripts\run_etch.ps1
```
- Note: This PowerShell script automatically handles directory switching to third_party/iTransformer.
- Mechanism: The model learns to forecast future sensor values. During inference, it exports the Attention Matrix (attention_weights.npy).
- Output: Saves model checkpoints and attention weights to the results/ folder.

### Step 3: Graph Construction & Visualization
Analyze the learned attention weights and generate topology graphs.
```bash
python main_visualize.py
```
- Logic: Automatically loads the latest experiment results.
- Output: Saves visualization figures (Heatmap, Signal Graph, Module Graph) to results/<experiment_id>/figures/.

## 📊 Visualization & Analysis

This project provides a comprehensive visualization pipeline to interpret the learned attention mechanisms from three different perspectives:

### 1. Attention Heatmap (Global View)
The raw attention matrix capturing the correlation intensity between all sensor pairs.
- **X/Y Axis**: All sensor features.
- **Color**: Brighter/Darker colors indicate higher attention weights (stronger dependencies).

![Attention Heatmap](figures/attn_heatmap.png)
*(Result from `main_visualize.py`)*

### 2. Signal Topology
A directed graph showing the critical paths between specific sensors.
- **Filtering**: Edges are filtered using the `Mean + n * Std` threshold to remove noise.
- **Insight**: Reveals specific sensor-to-sensor strong dependencies.

![Signal Graph](figures/signal_graph.png)

### 3. Module Topology (Coarse-grained)
A high-level system abstraction based on `sensor_mapping.json`.
- **Aggregation**: Sensors are grouped into functional modules.
- **Insight**: Displays the macroscopic interaction logic between different subsystems.

![Module Graph](figures/module_graph.png)