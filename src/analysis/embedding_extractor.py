"""
Utility for extracting and analyzing learned feature embeddings from iTransformer.

The iTransformer treats each sensor as a token (not each time step), so the
embedding layer learns a d_model-dimensional representation for each feature.

Usage::

    extractor = EmbeddingExtractor(model, device='cuda')
    embeddings = extractor.extract(data_loader)
    np.save('embeddings.npy', embeddings)
"""

import torch
import numpy as np
from typing import Optional


class EmbeddingExtractor:
    """Extract feature embeddings from a trained iTransformer model."""

    def __init__(self, model, device: str = 'cpu'):
        """
        :param model: Trained iTransformer model instance.
        :param device: 'cpu' or 'cuda'.
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def extract(self, data_loader, num_batches: Optional[int] = None) -> np.ndarray:
        """
        Extract embeddings by passing data through the model's embedding layer.

        :param data_loader: PyTorch DataLoader with test data.
        :param num_batches: Number of batches to process (None = all).
        :returns: Array of shape [N_features, d_model] — mean across all samples.
        """
        all_embeddings = []

        with torch.no_grad():
            for batch_idx, (batch_x, _, batch_x_mark, _) in enumerate(data_loader):
                if num_batches and batch_idx >= num_batches:
                    break

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Extract embeddings from the model's encoder
                # iTransformer input: [B, L, N] where N = num_features
                B, L, N = batch_x.shape

                # 1. 归一化 (与 iTransformer 模型 forward 过程保持一致)
                if getattr(self.model, 'use_norm', False):
                    means = batch_x.mean(1, keepdim=True).detach()
                    batch_x = batch_x - means
                    stdev = torch.sqrt(torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
                    batch_x /= stdev

                # 2. 直接传入 batch_x，不要外部转置！(DataEmbedding_inverted 内部会自动转置)
                # 输出的 enc_out 形状将是 [B, N + 时间协变量数量, d_model]
                enc_out = self.model.enc_embedding(batch_x, batch_x_mark)

                # 3. 截断掉末尾的时间协变量 tokens，只保留前 N 个物理传感器的 embeddings
                enc_out = enc_out[:, :N, :]

                all_embeddings.append(enc_out.cpu().numpy())

        # Concatenate all batches and average over batch dimension
        all_embeddings = np.concatenate(all_embeddings, axis=0)  # [Total_samples, N, d_model]
        mean_embeddings = np.mean(all_embeddings, axis=0)  # [N, d_model]

        print(f"✅ Extracted embeddings: {mean_embeddings.shape}")
        return mean_embeddings

    @staticmethod
    def load_model_from_checkpoint(checkpoint_path: str, model_class, args):
        """
        Helper to load a trained model from checkpoint.

        :param checkpoint_path: Path to .pth file.
        :param model_class: The model class (e.g., iTransformer.Model).
        :param args: The argparse args used during training.
        :returns: Loaded model instance.
        """
        model = model_class(args).float()
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"✅ Loaded model from: {checkpoint_path}")
        return model


def extract_embeddings_from_result_folder(
    result_folder: str,
    data_loader,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Convenience function to extract embeddings from a trained experiment.

    :param result_folder: Path to experiment result folder (contains checkpoint.pth).
    :param data_loader: DataLoader for inference.
    :param device: 'cpu' or 'cuda'.
    :returns: Feature embeddings [N, d_model].
    """
    import os
    import sys
    import torch

    # Dynamically import iTransformer model
    sys.path.append('./third_party/iTransformer')
    from model.iTransformer import Model

    # Find checkpoint
    checkpoint_path = os.path.join(result_folder, 'checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint.pth not found in: {result_folder}")

    # Load model (you'll need to reconstruct args from the experiment)
    # For simplicity, we assume args are saved or can be inferred
    # In practice, you should save args.pkl during training

    print("⚠️  You need to provide model args for loading.")
    print("    See example in main_extract_embeddings.py")

    raise NotImplementedError(
        "Please use the full workflow in main_extract_embeddings.py"
    )
