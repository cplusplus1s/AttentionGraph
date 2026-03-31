import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


# =========================================================================
# 新增模块：拓扑感知空间位置编码 (Graph Laplacian Spatial PE)
# =========================================================================
class TopologySpatialEmbedding(nn.Module):
    def __init__(self, num_nodes, d_model, alpha=0.1):
        super().__init__()

        # 1. 获取物理系统的健康基线邻接矩阵
        adj = self._build_healthy_adjacency(num_nodes)

        # 2. 计算归一化拉普拉斯矩阵 (Normalized Graph Laplacian)
        deg = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(deg, -0.5, out=np.zeros_like(deg), where=deg!=0)
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        laplacian = np.eye(num_nodes) - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

        # 3. 特征值分解 (Eigen Decomposition)
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        # 将特征向量作为空间位置编码的基底矩阵 [N, N]
        self.pe_tensor = torch.tensor(eigenvectors, dtype=torch.float32)

        # 4. 将 N 维的纯物理空间映射到 Transformer 的高维隐空间 d_model
        self.projection = nn.Linear(num_nodes, d_model)

        self.alpha = alpha

    def _build_healthy_adjacency(self, num_nodes):
        """
        基于 generalized_2D_msd.m 的真实物理受力拓扑，构建 24x24 的精确邻接矩阵。
        """
        # num_nodes 应为 24 (12 个质量块 * 2 个坐标维度)
        adj = np.zeros((num_nodes, num_nodes))

        # 辅助函数：将 1~12 的 Mass 编号映射为 0~23 的张量特征索引
        def get_idx(mass_id):
            return 2 * (mass_id - 1), 2 * (mass_id - 1) + 1

        # ---------------------------------------------------------
        # 步骤 1：枚举 3x4 网格中所有理论上存在的弹簧边
        # ---------------------------------------------------------
        edges_horizontal = []
        edges_vertical = []
        edges_diagonal = []

        for i in range(1, 4):     # Row: 1, 2, 3
            for j in range(1, 5): # Col: 1, 2, 3, 4
                u = (i - 1) * 4 + j

                if j < 4: edges_horizontal.append((u, u + 1))       # 右侧水平邻居
                if i < 3: edges_vertical.append((u, u + 4))         # 下方垂直邻居
                if i < 3 and j < 4: edges_diagonal.append((u, u + 5)) # 右下对角 (\)
                if i < 3 and j > 1: edges_diagonal.append((u, u + 3)) # 左下对角 (/)

        # ---------------------------------------------------------
        # 步骤 2：对齐 MATLAB 中被强行切断的弹簧 (continue 逻辑)
        # ---------------------------------------------------------
        # 严格提取自 generalized_2D_msd.m (包含物理上根本不相邻的无效限制)
        broken_pairs = {
            (1, 2), (2, 3), (5, 6), (8, 9), (11, 12),
            (7, 8), (10, 11), (4, 8), (6, 8), (7, 11),
            (8, 12), (1, 6), (3, 8), (2, 5)
        }

        def is_broken(u, v):
            return (u, v) in broken_pairs or (v, u) in broken_pairs

        # ---------------------------------------------------------
        # 步骤 3：力学定向映射 (Force Direction Mapping)
        # ---------------------------------------------------------

        # 1. 水平弹簧 (仅影响 X 轴)
        for u, v in edges_horizontal:
            if not is_broken(u, v):
                idxX_u, _ = get_idx(u)
                idxX_v, _ = get_idx(v)
                adj[idxX_u, idxX_v] = 1
                adj[idxX_v, idxX_u] = 1

        # 2. 垂直弹簧 (仅影响 Y 轴)
        for u, v in edges_vertical:
            if not is_broken(u, v):
                _, idxY_u = get_idx(u)
                _, idxY_v = get_idx(v)
                adj[idxY_u, idxY_v] = 1
                adj[idxY_v, idxY_u] = 1

        # 3. 对角线弹簧 (产生 X 和 Y 的多维耦合)
        for u, v in edges_diagonal:
            if not is_broken(u, v):
                idxX_u, idxY_u = get_idx(u)
                idxX_v, idxY_v = get_idx(v)
                # 相互投射力，形成 4x4 的全连通子块
                adj[idxX_u, idxX_v] = 1; adj[idxX_v, idxX_u] = 1
                adj[idxY_u, idxY_v] = 1; adj[idxY_v, idxY_u] = 1
                adj[idxX_u, idxY_v] = 1; adj[idxY_v, idxX_u] = 1
                adj[idxY_u, idxX_v] = 1; adj[idxX_v, idxY_u] = 1

        # 确保对角线没有自环 (拉普拉斯矩阵的定义要求)
        np.fill_diagonal(adj, 0)

        return adj

    def forward(self, x):
        # x shape: [Batch, N_total, d_model]  (N_total 可能大于 num_nodes, 因为有时间协变量)
        pe = self.projection(self.pe_tensor.to(x.device)) # [num_nodes, d_model]

        # 安全拷贝，避免就地修改 (In-place operation) 引发梯度计算报错
        x_new = x.clone()

        # 🚨 核心切片对齐：仅对前 num_nodes 个物理变量注入空间编码！
        num_physical_nodes = pe.shape[0]
        x_new[:, :num_physical_nodes, :] = x[:, :num_physical_nodes, :] + self.alpha * pe.unsqueeze(0)

        return x_new


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # 💡 [修改 1] 获取开关标志，兼容旧版本配置防报错
        self.use_spatial_pe = getattr(configs, 'use_spatial_pe', False)

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        if self.use_spatial_pe:
            # enc_in 对应变量总数 N (例如24)
            self.spatial_embedding = TopologySpatialEmbedding(configs.enc_in, configs.d_model)

        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens

        # 💡 [修改 3] 在数据输入 Encoder 前，注入物理空间拓扑！
        if self.use_spatial_pe:
            enc_out = self.spatial_embedding(enc_out)

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]