# AttentionGraph

A structure-aware fault diagnosis framework for complex multivariate systems, using the attention matrices of an iTransformer model as a proxy for the system's functional topology.

## Overview

The framework treats each sensor signal as a distinct token (the **variate-as-token** paradigm of iTransformer) and uses the resulting cross-signal attention matrix as a proxy for the system's functional topology. From this attention matrix, three diagnostic stages are constructed:

1. **Topology Estimation** — Reconstructs the system's functional connectivity by sparsifying the attention matrix and optionally aggregating into a module-level graph.

2. **Fault Detection** — Compares the test attention map against a healthy baseline using two complementary scoring functions: **Frobenius-norm drift** (aggregate per-edge deviation) and **spectral-gap drift** (Markov-chain mixing properties).

3. **Root Cause Analysis** — Identifies candidate root-cause signals using TokenRank topological drift, refines the ranking via expert-in-the-loop preference learning (CoExBO), and traces propagation paths through forward BFS on the attention difference matrix.

The framework is validated on:
- **Industrial dataset**: telemetry from a semiconductor manufacturing tool (29 active signals, 13 functional modules)
- **Brian2 SNN simulations**: four spiking neural network topologies (highway, chains, funnel, binary tree) with controlled synaptic-fault injection

## Repository structure

```
AttentionGraph/
├── brian2_simulator/             # Brian2 SNN simulation generators
│   ├── generate_brian2_sandbox.py        # Core simulator (LIF neurons, topology builder)
│   ├── generate_brian2_5x5_batch.py      # Batch generator for all topologies
│   ├── generate_brian2_highway_fault.py  # Fault scenario generator for highway
│   └── visualize_brian2.py               # Plots simulation outputs
│
├── config/                        # YAML configurations and sensor mappings
│   ├── settings_brian2.yaml             # SNN data processing config
│   └── sensor_mapping_*.json            # Sensor-to-module mappings
│
├── data/                          # Raw and processed data (not versioned)
│   ├── raw/                              # Original CSV files
│   └── processed/                        # Resampled, filtered, normalized data
│
├── src/                           # Framework source code
│   ├── etl/                              # Data loading and preprocessing
│   │   ├── base.py                       # Base preprocessor abstractions
│   │   ├── brian2_loader.py              # Brian2-specific loader
│   │   └── preprocessor.py               # Resampling, gap-fill, scaler persistence
│   │
│   ├── analysis/                         # Attention matrix extraction and visualization
│   │   ├── graph_builder.py              # Builds graphs from attention matrices
│   │   ├── data_exporter.py              # Exports adjacency CSVs
│   │   ├── embedding_extractor.py        # Extracts model embeddings
│   │   └── embedding_visualizer.py       # Visualizes embedding clusters
│   │
│   ├── diagnosis/                        # Fault detection and root cause analysis
│   │   ├── base.py                       # Base diagnoser class
│   │   ├── attention_drift.py            # Frobenius + spectral gap drift
│   │   ├── path_tracing.py               # Forward BFS propagation tracing
│   │   ├── coexbo_diagnoser.py           # Expert-in-the-loop preference learning
│   │   └── root_cause_analysis.py        # Aggregated root cause pipeline
│   │
│   └── visualization/                    # Plotting utilities
│
├── third_party/iTransformer/      # Patched iTransformer codebase
│   ├── experiments/
│   │   └── exp_long_term_forecasting.py  # Training and inference loop
│   ├── data_provider/
│   │   └── data_loader.py                # PATCHED: persistent StandardScaler
│   └── layers/                           # Core model architecture
│
├── run_scripts/                   # Pipeline orchestration
│   └── run_brian2.ps1                    # Train + healthy + faulty inference
│
├── results/                       # Outputs (not versioned)
│   ├── checkpoints/                      # Trained model weights
│   ├── healthy_baseline/                 # Healthy attention maps
│   └── unhealthy_test/                   # Faulty attention maps
│
├── main_pipeline.py               # ETL entry point (raw → processed CSVs)
├── main_visualize.py              # Generates attention graphs from .npy files
├── main_diagnosis.py              # Runs fault detection + root cause analysis
├── main_performance.py            # Computes forecasting metrics (MSE, MAE)
└── README.md                      # This file
```

## Installation

```bash
git clone https://github.com/cplusplus1s/AttentionGraph.git
cd AttentionGraph
conda env create -f environment.yml  # Or use requirements.txt
conda activate itransformer
```

Key dependencies:
- Python 3.10
- PyTorch (CUDA 11+ recommended)
- Brian2 (for SNN simulation)
- pandas, numpy, scikit-learn
- scipy (for spectral analysis)
- GPy or BoTorch (for CoExBO Gaussian Process preference learning)

## Quick start: end-to-end SNN experiment

### Step 1 — Generate Brian2 simulation data

```bash
# Generate 20 healthy runs + 20 faulty runs for all 4 topologies
python brian2_simulator/generate_brian2_5x5_batch.py

# Or generate only highway fault data with a specific severed edge
python brian2_simulator/generate_brian2_highway_fault.py
```

Output: `data/raw/brian2/{topology}/{healthy,unhealthy}_N/brian2_data.csv`

### Step 2 — Preprocess data

```bash
python main_pipeline.py
```

This applies:
- Uniform frequency resampling
- Gap filling (forward + back fill)
- Constant-column removal
- Multi-run concatenation with gap padding for training
- StandardScaler fitting and persistence

Output: `data/processed/{combined_healthy_train,brian2_healthy_N,brian2_unhealthy_N}.csv`

### Step 3 — Train iTransformer and run inference

```bash
# Windows PowerShell
.\run_scripts\run_brian2.ps1
```

This script runs three phases:
1. **Training**: Trains iTransformer on `combined_healthy_train.csv` (10 epochs, MSE loss)
2. **Healthy inference**: Runs inference on each `brian2_healthy_N.csv`, extracts attention maps
3. **Faulty inference**: Runs inference on each `brian2_unhealthy_N.csv`, extracts attention maps

Output: `results/checkpoints/` (model), `results/healthy_baseline/` and `results/unhealthy_test/` (attention maps)

### Step 4 — Generate topology graphs

```bash
python main_visualize.py --target_folder healthy_baseline
python main_visualize.py --target_folder unhealthy_test
```

Output: topology graphs (signal-level and module-level) saved alongside the attention maps.

### Step 5 — Run diagnosis

```bash
python main_diagnosis.py
```

This runs the full diagnostic pipeline:
- Frobenius drift detection
- Spectral gap drift detection
- TokenRank-based root cause ranking
- CoExBO expert-in-the-loop refinement
- Forward BFS propagation tracing

Output: console report of detection rates, z-scores, root cause candidates, and propagation paths.

## Key methodology

### Variate-as-token attention

The iTransformer treats each sensor's full time-series as a token, producing a row-stochastic attention matrix $\hat{\mathbf{A}} \in \mathbb{R}^{N \times N}$. We average attention across all heads and layers to obtain a robust proxy adjacency matrix that captures inter-signal functional dependencies.

### Frobenius drift detection

For a test attention map $\hat{\mathbf{A}}_\text{test}$ and a healthy baseline $\bar{\mathbf{A}}_\text{healthy}$:

$$S_\text{frob} = \| \hat{\mathbf{A}}_\text{test} - \bar{\mathbf{A}}_\text{healthy} \|_F$$

A z-score is computed against the leave-one-out null distribution of healthy runs; detection occurs when $z > z_\alpha$ (typically $z_\alpha = 2$ or $3$).

### Spectral gap drift detection

The spectral gap $g(\hat{\mathbf{A}}) = 1 - |\lambda_2|$ measures the mixing properties of the attention-induced Markov chain. Spectral gap drift is the absolute difference from baseline. This method is included as a theoretical comparison; in our experiments, Frobenius drift is more robust.

### TokenRank-based root cause ranking

TokenRank $\boldsymbol{\pi}$ is the stationary distribution of the row-normalized attention matrix (analogous to PageRank). Signals with the largest per-signal TokenRank drift are flagged as candidate root causes.

### Expert-in-the-loop refinement

Pairwise preferences from a domain expert (real or synthetic) train a Gaussian Process preference classifier. Soft-Copeland aggregation produces a prior score, which is fused with TokenRank drift via a Bayesian conjugate update.

### Forward BFS propagation tracing

The attention difference matrix $\mathbf{D}_{ij} = | \hat{\mathbf{A}}_\text{test}^{ij} - \bar{\mathbf{A}}_\text{healthy}^{ij} |$ is searched via depth-limited breadth-first search, expanding along edges with $D_{ij} > \tau$ to reconstruct downstream propagation paths from the top root cause candidate.

This work builds on:

- **iTransformer**: Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting", ICLR 2024 ([repo](https://github.com/thuml/iTransformer))
- **Brian2**: Stimberg et al., "Brian 2, an intuitive and efficient neural simulator", eLife 2019
