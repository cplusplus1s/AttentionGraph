"""
main_train_gnn.py — Train GNN fault classifier on synthetic data.

Creates synthetic faulty samples, trains a GCN classifier, and evaluates
on test data.

Usage::

    python main_train_gnn.py
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torch_geometric.data import DataLoader as GeometricDataLoader

from src.analysis.result_loader import ResultLoader, find_latest_result_folder, load_sensor_names
from src.diagnosis.gnn_classifier import (
    SyntheticFaultGenerator,
    FaultClassifierGNN,
    prepare_graph_data,
    train_gnn_classifier,
    predict_fault
)


def main():
    print("🚀 GNN Fault Classifier Training Pipeline\n")
    
    # 1. Load config
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    results_root = config['paths']['results_dir']
    data_path = config['paths']['processed_csv']
    
    # 2. Find latest result
    latest_folder = find_latest_result_folder(results_root)
    print(f"📂 Using result folder: {os.path.basename(latest_folder)}\n")
    
    # 3. Load attention matrix
    loader = ResultLoader(results_root, data_path)
    attention_matrix, feature_names = loader.load_data()
    
    # 4. Load normal time-series data
    print("📥 Loading normal time-series data...")
    df_full = pd.read_csv(data_path)
    feature_cols = [c for c in df_full.columns if c not in ['date', 'OT']]
    normal_data = df_full[feature_cols].values
    
    # Use a representative slice (e.g., first 2000 timesteps)
    normal_slice = normal_data[:2000]
    print(f"   Normal data shape: {normal_slice.shape}\n")
    
    # 5. Generate synthetic fault dataset
    print("🔄 Generating synthetic fault dataset...")
    generator = SyntheticFaultGenerator()
    fault_dataset = generator.generate_fault_dataset(
        normal_slice,
        num_samples_per_fault=30
    )
    
    print(f"   Total samples: {len(fault_dataset)}")
    print(f"   Fault types: {generator.FAULT_TYPES}\n")
    
    # 6. Convert to PyG Data objects
    print("🔄 Converting to graph representations...")
    graph_data_list = []
    
    for faulty_ts, label in fault_dataset:
        graph = prepare_graph_data(
            faulty_ts,
            attention_matrix,
            label,
            threshold_std=1.5
        )
        graph_data_list.append(graph)
    
    print(f"✅ Created {len(graph_data_list)} graph objects\n")
    
    # 7. Train/val/test split
    train_graphs, test_graphs = train_test_split(
        graph_data_list, test_size=0.2, random_state=42
    )
    train_graphs, val_graphs = train_test_split(
        train_graphs, test_size=0.2, random_state=42
    )
    
    print(f"Split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}\n")
    
    # 8. Create data loaders
    train_loader = GeometricDataLoader(train_graphs, batch_size=16, shuffle=True)
    val_loader = GeometricDataLoader(val_graphs, batch_size=16, shuffle=False)
    test_loader = GeometricDataLoader(test_graphs, batch_size=1, shuffle=False)
    
    # 9. Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    num_node_features = 5  # [mean, std, min, max, skew]
    num_fault_types = len(generator.FAULT_TYPES)
    
    model = train_gnn_classifier(
        train_loader, val_loader,
        num_node_features=num_node_features,
        num_fault_types=num_fault_types,
        num_epochs=100,
        lr=0.001,
        device=device
    )
    
    # 10. Test evaluation
    print("\n" + "="*70)
    print("📊 Test Set Evaluation")
    print("="*70)
    
    model.eval()
    test_correct = 0
    test_total = 0
    
    confusion_matrix = np.zeros((num_fault_types, num_fault_types), dtype=int)
    
    for data in test_loader:
        data = data.to(device)
        pred, probs = predict_fault(model, data, device)
        true_label = data.y.item()
        
        confusion_matrix[true_label, pred] += 1
        if pred == true_label:
            test_correct += 1
        test_total += 1
    
    test_acc = test_correct / test_total
    print(f"\n🎯 Test Accuracy: {test_acc:.3f} ({test_correct}/{test_total})\n")
    
    # Print confusion matrix
    print("Confusion Matrix:")
    print("=" * 70)
    header = "True\\Pred | " + " | ".join(f"{ft[:6]:>6s}" for ft in generator.FAULT_TYPES)
    print(header)
    print("-" * 70)
    
    for i, true_ft in enumerate(generator.FAULT_TYPES):
        row = f"{true_ft[:8]:<10} | " + " | ".join(f"{confusion_matrix[i,j]:>6d}" for j in range(num_fault_types))
        print(row)
    
    # 11. Save model
    model_save_path = os.path.join(latest_folder, "gnn_fault_classifier.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\n💾 Model saved: {model_save_path}")
    
    # 12. Demo: Predict on a few test samples
    print("\n" + "="*70)
    print("🔍 Sample Predictions")
    print("="*70)
    
    for i, data in enumerate(test_loader):
        if i >= 5:  # Show first 5
            break
        
        data = data.to(device)
        pred, probs = predict_fault(model, data, device)
        true_label = data.y.item()
        
        print(f"\nSample {i+1}:")
        print(f"  True: {generator.FAULT_TYPES[true_label]}")
        print(f"  Pred: {generator.FAULT_TYPES[pred]}")
        print(f"  Confidence: {probs[pred]:.3f}")
        print(f"  Top-3 probs: {', '.join([f'{generator.FAULT_TYPES[j]}:{probs[j]:.3f}' for j in np.argsort(probs)[-3:][::-1]])}")
    
    print("\n" + "="*70)
    print("✅ GNN Training Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
