"""
Graph Neural Network for fault classification.

Trains a GCN to classify different fault types using attention-derived graphs
as input. Includes synthetic fault generator for training data creation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader as GeometricDataLoader
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Synthetic Fault Generator
# ============================================================================

class SyntheticFaultGenerator:
    """
    Generate synthetic faulty data for training GNN classifier.
    
    Fault types:
    1. Sensor bias (constant offset)
    2. Sensor drift (increasing/decreasing trend)
    3. Sensor noise (increased variance)
    4. Sensor stuck (constant value)
    5. Coupling fault (one sensor affects another abnormally)
    """
    
    FAULT_TYPES = [
        'normal',
        'bias',
        'drift',
        'noise',
        'stuck',
        'coupling'
    ]
    
    @staticmethod
    def inject_bias_fault(data: np.ndarray, sensor_idx: int,
                          magnitude: float = 2.0) -> np.ndarray:
        """Add constant offset to a sensor."""
        faulty_data = data.copy()
        faulty_data[:, sensor_idx] += magnitude
        return faulty_data
    
    @staticmethod
    def inject_drift_fault(data: np.ndarray, sensor_idx: int,
                           drift_rate: float = 0.01) -> np.ndarray:
        """Add linearly increasing trend to a sensor."""
        faulty_data = data.copy()
        T = len(faulty_data)
        trend = np.arange(T) * drift_rate
        faulty_data[:, sensor_idx] += trend
        return faulty_data
    
    @staticmethod
    def inject_noise_fault(data: np.ndarray, sensor_idx: int,
                           noise_scale: float = 0.5) -> np.ndarray:
        """Increase noise variance of a sensor."""
        faulty_data = data.copy()
        noise = np.random.randn(len(faulty_data)) * noise_scale
        faulty_data[:, sensor_idx] += noise
        return faulty_data
    
    @staticmethod
    def inject_stuck_fault(data: np.ndarray, sensor_idx: int,
                          stuck_time: int = None) -> np.ndarray:
        """Freeze sensor at a constant value after stuck_time."""
        faulty_data = data.copy()
        if stuck_time is None:
            stuck_time = len(faulty_data) // 3
        
        stuck_value = faulty_data[stuck_time, sensor_idx]
        faulty_data[stuck_time:, sensor_idx] = stuck_value
        return faulty_data
    
    @staticmethod
    def inject_coupling_fault(data: np.ndarray, source_idx: int,
                              target_idx: int, coupling_strength: float = 0.5) -> np.ndarray:
        """Make target sensor abnormally coupled to source sensor."""
        faulty_data = data.copy()
        faulty_data[:, target_idx] += coupling_strength * faulty_data[:, source_idx]
        return faulty_data
    
    @classmethod
    def generate_fault_dataset(cls, normal_data: np.ndarray,
                               num_samples_per_fault: int = 20) -> List[Tuple[np.ndarray, int]]:
        """
        Generate a balanced dataset of normal + faulty samples.
        
        :param normal_data: Normal time-series [T, N].
        :param num_samples_per_fault: How many examples per fault type.
        :returns: List of (faulty_data, label) tuples.
        """
        dataset = []
        n_sensors = normal_data.shape[1]
        
        # Normal samples
        for _ in range(num_samples_per_fault):
            dataset.append((normal_data.copy(), 0))  # Label 0 = normal
        
        # Bias faults
        for _ in range(num_samples_per_fault):
            sensor_idx = np.random.randint(0, n_sensors)
            magnitude = np.random.uniform(1.0, 3.0)
            faulty = cls.inject_bias_fault(normal_data, sensor_idx, magnitude)
            dataset.append((faulty, 1))
        
        # Drift faults
        for _ in range(num_samples_per_fault):
            sensor_idx = np.random.randint(0, n_sensors)
            drift_rate = np.random.uniform(0.005, 0.02)
            faulty = cls.inject_drift_fault(normal_data, sensor_idx, drift_rate)
            dataset.append((faulty, 2))
        
        # Noise faults
        for _ in range(num_samples_per_fault):
            sensor_idx = np.random.randint(0, n_sensors)
            noise_scale = np.random.uniform(0.3, 1.0)
            faulty = cls.inject_noise_fault(normal_data, sensor_idx, noise_scale)
            dataset.append((faulty, 3))
        
        # Stuck faults
        for _ in range(num_samples_per_fault):
            sensor_idx = np.random.randint(0, n_sensors)
            stuck_time = np.random.randint(len(normal_data) // 4, len(normal_data) // 2)
            faulty = cls.inject_stuck_fault(normal_data, sensor_idx, stuck_time)
            dataset.append((faulty, 4))
        
        # Coupling faults
        for _ in range(num_samples_per_fault):
            source_idx = np.random.randint(0, n_sensors)
            target_idx = np.random.randint(0, n_sensors)
            while target_idx == source_idx:
                target_idx = np.random.randint(0, n_sensors)
            coupling = np.random.uniform(0.3, 0.7)
            faulty = cls.inject_coupling_fault(normal_data, source_idx, target_idx, coupling)
            dataset.append((faulty, 5))
        
        print(f"✅ Generated {len(dataset)} synthetic fault samples")
        return dataset


# ============================================================================
# GNN Model
# ============================================================================

class FaultClassifierGNN(nn.Module):
    """
    Graph Convolutional Network for fault classification.
    
    Architecture:
    - Input: Node features (sensor statistics) + Graph structure (attention edges)
    - 2 GCN layers with ReLU
    - Global mean pooling
    - Linear classifier
    """
    
    def __init__(self, num_node_features: int, num_fault_types: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.fc = nn.Linear(hidden_dim // 2, num_fault_types)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        :param x: Node features [N, F].
        :param edge_index: Edge connectivity [2, E].
        :param edge_weight: Optional edge weights [E].
        :param batch: Batch assignment for each node (for batched graphs).
        :returns: Class logits [B, num_classes] if batched, else [1, num_classes].
        """
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        
        # Global pooling
        if batch is not None:
            # Batched graph: pool per-graph
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)
        else:
            # Single graph: mean across all nodes
            x = x.mean(dim=0, keepdim=True)
        
        # Classification
        x = self.fc(x)
        return x


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_graph_data(
    faulty_timeseries: np.ndarray,
    attention_matrix: np.ndarray,
    label: int,
    threshold_std: float = 1.5
) -> Data:
    """
    Convert a faulty time-series + attention matrix into a PyG Data object.
    
    Node features: [mean, std, min, max, skew] of each sensor's time-series.
    
    :param faulty_timeseries: [T, N] time-series data.
    :param attention_matrix: [N, N] attention weights.
    :param label: Fault type label.
    :param threshold_std: Edge threshold (mean + k*std).
    :returns: PyG Data object.
    """
    from scipy.stats import skew
    
    T, N = faulty_timeseries.shape
    
    # 1. Compute node features (statistical summary of each sensor)
    node_features = []
    for i in range(N):
        sensor_data = faulty_timeseries[:, i]
        features = [
            np.mean(sensor_data),
            np.std(sensor_data),
            np.min(sensor_data),
            np.max(sensor_data),
            skew(sensor_data)
        ]
        node_features.append(features)
    
    node_features = np.array(node_features, dtype=np.float32)  # [N, 5]
    
    # 2. Build edge index from attention matrix (threshold high-attention edges)
    threshold = np.mean(attention_matrix) + threshold_std * np.std(attention_matrix)
    
    edge_index = []
    edge_weight = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if attention_matrix[i, j] > threshold:
                edge_index.append([j, i])  # j -> i (source -> target)
                edge_weight.append(attention_matrix[i, j])
    
    if not edge_index:
        # No edges pass threshold; add self-loops to avoid empty graph
        edge_index = [[i, i] for i in range(N)]
        edge_weight = [1.0] * N
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    
    # 3. Create PyG Data object
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)


# ============================================================================
# Training Loop
# ============================================================================

def train_gnn_classifier(
    train_loader: GeometricDataLoader,
    val_loader: GeometricDataLoader,
    num_node_features: int,
    num_fault_types: int,
    num_epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu'
) -> FaultClassifierGNN:
    """
    Train the GNN fault classifier.
    
    :param train_loader: PyG DataLoader for training.
    :param val_loader: PyG DataLoader for validation.
    :param num_node_features: Dimension of node features (e.g., 5).
    :param num_fault_types: Number of fault classes.
    :param num_epochs: Training epochs.
    :param lr: Learning rate.
    :param device: 'cpu' or 'cuda'.
    :returns: Trained model.
    """
    model = FaultClassifierGNN(num_node_features, num_fault_types).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_model_state = None
    
    print("🚀 Training GNN Fault Classifier...")
    print("="*60)
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)
        
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"Train Acc: {train_acc:.3f} | "
                  f"Val Acc: {val_acc:.3f}")
    
    print("="*60)
    print(f"✅ Training complete. Best Val Acc: {best_val_acc:.3f}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model


def predict_fault(model: FaultClassifierGNN, graph_data: Data,
                  device: str = 'cpu') -> Tuple[int, np.ndarray]:
    """
    Predict fault type for a single graph.
    
    :param model: Trained GNN model.
    :param graph_data: PyG Data object.
    :param device: 'cpu' or 'cuda'.
    :returns: (predicted_label, class_probabilities).
    """
    model.eval()
    graph_data = graph_data.to(device)
    
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        pred = out.argmax(dim=1).item()
    
    return pred, probs
