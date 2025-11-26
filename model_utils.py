"""
TimeGAN Model Utilities - PyTorch Implementation

Utility functions for loading saved models and generating new data.
"""

import torch
import numpy as np
from timegan import Embedder, Recovery, Generator, Supervisor, Discriminator
from utils import random_generator


def load_timegan_model(model_path, device):
    """
    加载保存的 TimeGAN 模型

    Args:
        model_path: 模型文件路径
        device: torch.device('cuda') 或 torch.device('cpu')

    Returns:
        model_dict: 包含所有网络和参数的字典
    """
    model_dict = torch.load(model_path, map_location=device)
    print(f'✅ Model loaded from: {model_path}')

    # 打印模型信息
    params = model_dict['parameters']
    print(f'Model Parameters:')
    print(f'  - Hidden Dim: {params["hidden_dim"]}')
    print(f'  - Num Layers: {params["num_layers"]}')
    print(f'  - Module: {params["module_name"]}')
    print(f'  - Dimension: {params["dim"]}')

    return model_dict


def initialize_networks_from_dict(model_dict, device):
    """
    从模型字典初始化网络

    Args:
        model_dict: 加载的模型字典
        device: 设备

    Returns:
        embedder, recovery, generator, supervisor: 初始化的网络
    """
    params = model_dict['parameters']

    # 初始化网络
    embedder = Embedder(params['dim'], params['hidden_dim'],
                        params['num_layers'], params['module_name']).to(device)
    recovery = Recovery(params['hidden_dim'], params['dim'],
                        params['num_layers'], params['module_name']).to(device)
    generator = Generator(params['z_dim'], params['hidden_dim'],
                          params['num_layers'], params['module_name']).to(device)
    supervisor = Supervisor(params['hidden_dim'], params['num_layers'],
                            params['module_name']).to(device)

    # 加载权重
    embedder.load_state_dict(model_dict['embedder'])
    recovery.load_state_dict(model_dict['recovery'])
    generator.load_state_dict(model_dict['generator'])
    supervisor.load_state_dict(model_dict['supervisor'])

    # 设置为评估模式
    embedder.eval()
    recovery.eval()
    generator.eval()
    supervisor.eval()

    return embedder, recovery, generator, supervisor


def generate_synthetic_data(model_dict, num_samples, seq_lengths, device):
    """
    使用加载的模型生成新的合成数据

    Args:
        model_dict: 加载的模型字典
        num_samples: 要生成的样本数量
        seq_lengths: 每个样本的序列长度列表
        device: 设备

    Returns:
        generated_data: 生成的合成数据
    """
    params = model_dict['parameters']
    norm = model_dict['normalization']

    # 初始化网络
    generator, recovery, supervisor = initialize_networks_from_dict(model_dict, device)[2:]

    with torch.no_grad():
        # 生成随机向量
        Z_mb = random_generator(num_samples, params['z_dim'],
                                seq_lengths, params['max_seq_len'])
        Z_mb = torch.FloatTensor(np.array(Z_mb)).to(device)
        seq_lengths_tensor = torch.LongTensor(seq_lengths).cpu()

        # 生成数据
        E_hat = generator(Z_mb, seq_lengths_tensor)
        H_hat = supervisor(E_hat, seq_lengths_tensor)
        generated_data_curr = recovery(H_hat, seq_lengths_tensor)
        generated_data_curr = generated_data_curr.cpu().numpy()

    # 提取有效数据
    generated_data = []
    for i in range(num_samples):
        temp = generated_data_curr[i, :seq_lengths[i], :]
        generated_data.append(temp)

    # 反归一化
    generated_data = np.array(generated_data)
    generated_data = generated_data * norm['max_val']
    generated_data = generated_data + norm['min_val']

    print(f'✅ Generated {num_samples} synthetic samples')

    return generated_data


# 使用示例
if __name__ == '__main__':
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './saved_models/timegan_stock_20250120_071723.pt'

    model_dict = load_timegan_model(model_path, device)

    # 生成新数据
    num_samples = 1000
    seq_lengths = [24] * num_samples  # 所有样本长度为 24

    new_data = generate_synthetic_data(model_dict, num_samples, seq_lengths, device)

    print(f'Generated data shape: {np.array(new_data).shape}')