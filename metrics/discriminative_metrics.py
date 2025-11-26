"""Time-series Generative Adversarial Networks (TimeGAN) Codebase - PyTorch Implementation

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: 2025-10-17
Converted to PyTorch by: GitHub Copilot

-----------------------------

discriminative_metrics.py (PyTorch Version)

Note: Use post-hoc RNN to classify original data and synthetic data

Output: discriminative score (np.abs(classification accuracy - 0.5))
"""

# Necessary Packages
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from utils import train_test_divide, extract_time, batch_generator


class Discriminator(nn.Module):
    """Simple discriminator network."""

    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed_input)
        # Use last hidden state
        y_hat = self.fc(h_n.squeeze(0))
        return y_hat


def discriminative_score_metrics(ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data

    Args:
        - ori_data: original data (原始尺度，未归一化)
        - generated_data: generated synthetic data (原始尺度，未归一化)

    Returns:
        - discriminative_score: np.abs(classification accuracy - 0.5)
    """

    # ========== 添加数据验证 ==========
    print(f'\n🔍 Discriminative Metrics - 数据检查:')

    ori_array = np.array([d for d in ori_data])
    gen_array = np.array([d for d in generated_data])

    print(f'  原始数据范围: [{ori_array.min():.4f}, {ori_array.max():.4f}]')
    print(f'  生成数据范围: [{gen_array.min():.4f}, {gen_array.max():.4f}]')

    # 检查异常值
    if np.isnan(gen_array).any() or np.isinf(gen_array).any():
        print(f'  ❌ 错误: 生成数据包含 NaN 或 Inf!')
        return 0.5  # 返回 0.5 表示完全无法区分（最差情况）

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    # ========== 归一化数据（用于判别器训练）==========
    def normalize_data(data):
        """归一化数据"""
        data_array = np.array([d for d in data])
        min_val = data_array.min(axis=(0, 1))
        max_val = data_array.max(axis=(0, 1))

        normalized = []
        for d in data:
            norm_d = (d - min_val) / (max_val - min_val + 1e-7)
            normalized.append(norm_d)

        return normalized

    # 合并所有数据进行统一归一化
    all_data = list(ori_data) + list(generated_data)
    all_data_norm = normalize_data(all_data)

    # 分离归一化后的数据
    ori_data_norm = all_data_norm[:len(ori_data)]
    generated_data_norm = all_data_norm[len(ori_data):]

    print(f'  ✅ 数据已归一化用于判别器训练')

    # Network parameters
    hidden_dim = int(dim / 2)
    iterations = 2000
    batch_size = 128

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build discriminator
    discriminator = Discriminator(dim, hidden_dim).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(discriminator.parameters())

    # Train/test division (使用归一化后的数据)
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data_norm, generated_data_norm, ori_time, generated_time)

    # ... (训练代码保持不变，使用归一化后的数据) ...

    # Training
    discriminator.train()
    for itt in range(iterations):
        # Batch setting
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)

        # Convert to tensors
        X_mb = torch.FloatTensor(np.array(X_mb)).to(device)
        X_hat_mb = torch.FloatTensor(np.array(X_hat_mb)).to(device)
        T_mb = torch.LongTensor(T_mb).cpu()
        T_hat_mb = torch.LongTensor(T_hat_mb).cpu()

        # Labels
        y_real = torch.ones(len(X_mb), 1).to(device)
        y_fake = torch.zeros(len(X_hat_mb), 1).to(device)

        # Forward pass
        y_pred_real = discriminator(X_mb, T_mb)
        y_pred_fake = discriminator(X_hat_mb, T_hat_mb)

        # Compute loss
        d_loss_real = criterion(y_pred_real, y_real)
        d_loss_fake = criterion(y_pred_fake, y_fake)
        d_loss = d_loss_real + d_loss_fake

        # Backward and optimize
        optimizer.zero_grad()
        d_loss.backward()
        optimizer.step()

    # Test the performance on the testing set
    discriminator.eval()
    with torch.no_grad():
        # Convert test data to tensors
        test_x_tensor = torch.FloatTensor(np.array(test_x)).to(device)
        test_x_hat_tensor = torch.FloatTensor(np.array(test_x_hat)).to(device)
        test_t_tensor = torch.LongTensor(test_t).cpu()
        test_t_hat_tensor = torch.LongTensor(test_t_hat).cpu()

        # Predictions
        y_pred_real_curr = torch.sigmoid(discriminator(test_x_tensor, test_t_tensor))
        y_pred_fake_curr = torch.sigmoid(discriminator(test_x_hat_tensor, test_t_hat_tensor))

    # Convert to numpy
    y_pred_real_curr = y_pred_real_curr.cpu().numpy()
    y_pred_fake_curr = y_pred_fake_curr.cpu().numpy()

    y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0))
    y_label_final = np.concatenate((np.ones([len(y_pred_real_curr), ]),
                                    np.zeros([len(y_pred_fake_curr), ])), axis=0)

    # Compute the accuracy
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)

    print(f'  📊 Discriminative Score: {discriminative_score:.4f}')

    return discriminative_score