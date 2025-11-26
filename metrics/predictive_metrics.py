"""Time-series Generative Adversarial Networks (TimeGAN) Codebase - PyTorch Implementation

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: 2025-10-17
Converted to PyTorch by: GitHub Copilot

-----------------------------

predictive_metrics.py (PyTorch Version)

Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""

# Necessary Packages
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from utils import extract_time


class Predictor(nn.Module):
    """Simple predictor network."""

    def __init__(self, input_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Apply fully connected layer
        y_hat = self.sigmoid(self.fc(output))
        return y_hat


def predictive_score_metrics(ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction.

    Args:
        - ori_data: original data (原始尺度，未归一化)
        - generated_data: generated synthetic data (原始尺度，未归一化)

    Returns:
        - predictive_score: MAE of the predictions on the original data
    """

    # ========== 添加数据验证 ==========
    print(f'\n🔍 Predictive Metrics - 数据检查:')

    # 检查数据范围
    ori_array = np.array([d for d in ori_data])
    gen_array = np.array([d for d in generated_data])

    print(f'  原始数据范围: [{ori_array.min():.4f}, {ori_array.max():.4f}]')
    print(f'  生成数据范围: [{gen_array.min():.4f}, {gen_array.max():.4f}]')
    print(f'  原始数据均值: {ori_array.mean():.4f}')
    print(f'  生成数据均值: {gen_array.mean():.4f}')

    # 检查异常值
    if np.isnan(gen_array).any() or np.isinf(gen_array).any():
        print(f'  ❌ 错误: 生成数据包含 NaN 或 Inf，无法进行评估!')
        return float('inf')

    if abs(gen_array.max()) > 1e8 or abs(gen_array.min()) > 1e8:
        print(f'  ⚠️  警告: 生成数据数值异常，评估结果可能不准确!')

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    # Network parameters
    hidden_dim = int(dim / 2)
    iterations = 5000
    batch_size = 128

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== 对数据进行归一化（用于训练预测器）==========
    # 注意: 这里需要归一化，因为神经网络训练需要标准化的输入
    def normalize_data(data):
        """归一化数据用于神经网络训练"""
        data_array = np.array([d for d in data])
        min_val = data_array.min(axis=(0, 1))
        max_val = data_array.max(axis=(0, 1))

        normalized = []
        for d in data:
            norm_d = (d - min_val) / (max_val - min_val + 1e-7)
            normalized.append(norm_d)

        return normalized, min_val, max_val

    # 归一化训练数据（generated_data）
    generated_data_norm, gen_min, gen_max = normalize_data(generated_data)

    # 归一化测试数据（ori_data）- 使用相同的归一化参数
    ori_data_norm = []
    for d in ori_data:
        norm_d = (d - gen_min) / (gen_max - gen_min + 1e-7)
        ori_data_norm.append(norm_d)

    print(f'  ✅ 数据已归一化用于预测器训练')

    # Build predictor
    predictor = Predictor(dim - 1, hidden_dim).to(device)

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(predictor.parameters())

    # Training using Synthetic dataset (使用归一化的数据)
    predictor.train()
    for itt in range(iterations):
        # Set mini-batch
        idx = np.random.permutation(len(generated_data_norm))
        train_idx = idx[:batch_size]

        X_mb = list(generated_data_norm[i][:-1, :(dim - 1)] for i in train_idx)
        T_mb = list(generated_time[i] - 1 for i in train_idx)
        Y_mb = list(np.reshape(generated_data_norm[i][1:, (dim - 1)],
                               [len(generated_data_norm[i][1:, (dim - 1)]), 1]) for i in train_idx)

        # Convert to tensors
        X_mb = torch.FloatTensor(np.array(X_mb)).to(device)
        Y_mb = torch.FloatTensor(np.array(Y_mb)).to(device)
        T_mb = torch.LongTensor(T_mb).cpu()

        # Forward pass
        y_pred = predictor(X_mb, T_mb)

        # Compute loss
        p_loss = criterion(y_pred, Y_mb)

        # Backward and optimize
        optimizer.zero_grad()
        p_loss.backward()
        optimizer.step()

    # Test the trained model on the original data (使用归一化的数据)
    predictor.eval()
    with torch.no_grad():
        idx = np.random.permutation(len(ori_data_norm))
        train_idx = idx[:no]

        X_mb = list(ori_data_norm[i][:-1, :(dim - 1)] for i in train_idx)
        T_mb = list(ori_time[i] - 1 for i in train_idx)
        Y_mb = list(np.reshape(ori_data_norm[i][1:, (dim - 1)],
                               [len(ori_data_norm[i][1:, (dim - 1)]), 1]) for i in train_idx)

        # Convert to tensors
        X_mb = torch.FloatTensor(np.array(X_mb)).to(device)
        T_mb = torch.LongTensor(T_mb).cpu()

        # Prediction
        pred_Y_curr = predictor(X_mb, T_mb)
        pred_Y_curr = pred_Y_curr.cpu().numpy()

    # Compute the performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i, :, :])

    predictive_score = MAE_temp / no

    print(f'  📊 Predictive Score: {predictive_score:.4f}')

    return predictive_score