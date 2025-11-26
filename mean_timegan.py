"""Time-series Generative Adversarial Networks (TimeGAN) Codebase - PyTorch Implementation

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: 2025-10-20
Converted to PyTorch by: GitHub Copilot
Updated by: 123adk - Added model and data saving functionality

-----------------------------

main_timegan_experiment.py (PyTorch Version with Save Functionality)

(1) Import data
(2) Generate synthetic data
(3) Save trained model and generated data
(4) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import warnings
import torch
import os
from datetime import datetime
import json

warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization


def save_model_and_data(model_dict, generated_data, ori_data, args, metric_results, save_dir='./saved_models/wave'):
    """
    保存训练好的模型、生成的数据和评估结果

    Args:
        - model_dict: 模型字典（包含所有网络的state_dict）
        - generated_data: 生成的合成数据
        - ori_data: 原始数据
        - args: 命令行参数
        - metric_results: 评估指标结果
        - save_dir: 保存目录

    Returns:
        - saved_files: 保存的文件路径字典
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 生成时间戳和模型名称
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f'timegan_{args.data_name}_{timestamp}'

    # 定义保存路径
    model_save_path = os.path.join(save_dir, f'{model_name}.pt')
    generated_data_path = os.path.join(save_dir, f'{model_name}_generated_data')
    original_data_path = os.path.join(save_dir, f'{model_name}_original_data')
    params_save_path = os.path.join(save_dir, f'{model_name}_parameters.json')
    metrics_save_path = os.path.join(save_dir, f'{model_name}_metrics.json')

    print('\n' + '=' * 70)
    print('💾 保存模型和数据...')
    print('=' * 70)

    # 1. 保存模型
    try:
        torch.save(model_dict, model_save_path)
        print(f'✅ 模型已保存: {model_save_path}')
        model_size = os.path.getsize(model_save_path) / (1024 * 1024)  # MB
        print(f'   文件大小: {model_size:.2f} MB')
    except Exception as e:
        print(f'❌ 模型保存失败: {str(e)}')

    # 2. 保存生成的数据
    try:
        np.save(generated_data_path, generated_data)
        print(f'✅ 生成数据已保存: {generated_data_path}')
        print(f'   数据形状: {np.array(generated_data).shape}')
        data_size = os.path.getsize(generated_data_path) / (1024 * 1024)  # MB
        print(f'   文件大小: {data_size:.2f} MB')
    except Exception as e:
        print(f'❌ 生成数据保存失败: {str(e)}')

    # 3. 保存原始数据（用于后续对比）
    try:
        np.save(original_data_path, ori_data)
        print(f'✅ 原始数据已保存: {original_data_path}')
        print(f'   数据形状: {np.array(ori_data).shape}')
    except Exception as e:
        print(f'❌ 原始数据保存失败: {str(e)}')

    # 4. 保存训练参数
    try:
        params_dict = {
            'data_name': args.data_name,
            'seq_len': args.seq_len,
            'module': args.module,
            'hidden_dim': args.hidden_dim,
            'num_layer': args.num_layer,
            'iterations': args.iteration,
            'batch_size': args.batch_size,
            'metric_iteration': args.metric_iteration,
            'timestamp': timestamp,
            'user': '123adk',
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU'
        }

        with open(params_save_path, 'w') as f:
            json.dump(params_dict, f, indent=4)
        print(f'✅ 训练参数已保存: {params_save_path}')
    except Exception as e:
        print(f'❌ 参数保存失败: {str(e)}')

    # 5. 保存评估指标
    try:
        metrics_dict = {
            'discriminative_score': float(metric_results['discriminative']),
            'predictive_score': float(metric_results['predictive']),
            'timestamp': timestamp,
            'data_name': args.data_name
        }

        with open(metrics_save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f'✅ 评估指标已保存: {metrics_save_path}')
    except Exception as e:
        print(f'❌ 评估指标保存失败: {str(e)}')

    print('=' * 70)
    print('✅ 所有文件保存完成!')
    print('=' * 70)

    # 返回保存的文件路径
    saved_files = {
        'model': model_save_path,
        'generated_data': generated_data_path,
        'original_data': original_data_path,
        'parameters': params_save_path,
        'metrics': metrics_save_path,
        'model_name': model_name
    }

    return saved_files


def print_summary(saved_files, metric_results, args):
    """
    打印实验总结

    Args:
        - saved_files: 保存的文件路径字典
        - metric_results: 评估指标
        - args: 命令行参数
    """
    print('\n' + '=' * 70)
    print('📊 实验总结')
    print('=' * 70)
    print(f'实验名称: {saved_files["model_name"]}')
    print(f'数据集: {args.data_name}')
    print(f'序列长度: {args.seq_len}')
    print(f'训练迭代次数: {args.iteration}')
    print(f'批次大小: {args.batch_size}')
    print(f'\n📈 评估指标:')
    print(f'  • Discriminative Score: {metric_results["discriminative"]:.4f}')
    print(f'  • Predictive Score: {metric_results["predictive"]:.4f}')
    print(f'\n📁 保存的文件:')
    for file_type, file_path in saved_files.items():
        if file_type != 'model_name':
            print(f'  • {file_type}: {file_path}')
    print('=' * 70)


def main(args):
    """Main function for TimeGAN experiments.

    Args:
        - data_name: sine, stock, or energy
        - seq_len: sequence length
        - Network parameters (should be optimized for different datasets)
          - module: gru, lstm, or lstmLN
          - hidden_dim: hidden dimensions
          - num_layer: number of layers
          - iteration: number of training iterations
          - batch_size: the number of samples in each batch
        - metric_iteration: number of iterations for metric computation

    Returns:
        - ori_data: original data
        - generated_data: generated synthetic data
        - metric_results: discriminative and predictive scores
        - saved_files: paths to saved files
    """

    print('=' * 70)
    print('🚀 TimeGAN 实验开始')
    print('=' * 70)
    ori_data=np.load('data/Z_train.npy')
    # ==================== 1. Data loading ====================
    print('\n📂 加载数据...')
    #if args.data_name in ['stock', 'energy','wave']:
    #    ori_data = real_data_loading(args.data_name, args.seq_len)
    #elif args.data_name == 'sine':
        # Set number of samples and its dimensions
    #    no, dim = 10000, 5
    #    ori_data = sine_data_generation(no, args.seq_len, dim)

    print(f'✅ {args.data_name} 数据集已加载')
    print(f'   数据形状: {np.array(ori_data).shape}')

    # ==================== 2. Synthetic data generation ====================
    print('\n🔧 配置网络参数...')
    # Set network parameters
    parameters = dict()
    parameters['module'] = args.module
    parameters['hidden_dim'] = args.hidden_dim
    parameters['num_layer'] = args.num_layer
    parameters['iterations'] = args.iteration
    parameters['batch_size'] = args.batch_size

    # 添加设备配置（优先使用 GPU）
    if torch.cuda.is_available():
        parameters['device'] = torch.device('cuda')
        print(f'✅ 使用 GPU: {torch.cuda.get_device_name(0)}')
    else:
        parameters['device'] = torch.device('cpu')
        print('⚠️  使用 CPU (建议使用 GPU 以加速训练)')

    print(f'\n⏳ 开始训练 TimeGAN...')
    print(f'   模块类型: {args.module}')
    print(f'   隐藏层维度: {args.hidden_dim}')
    print(f'   网络层数: {args.num_layer}')
    print(f'   迭代次数: {args.iteration}')
    print(f'   批次大小: {args.batch_size}')

    # 训练 TimeGAN 并获取生成的数据和模型
    generated_data, model_dict = timegan(ori_data, parameters)
    print('✅ TimeGAN 训练完成!')

    # ==================== 3. Save model and data ====================
    # Performance metrics
    # Output initialization
    metric_results = dict()

    # 先进行评估，再保存（这样可以把评估结果一起保存）
    print('\n📊 评估生成数据质量...')

    # 1. Discriminative Score
    print(f'\n  计算 Discriminative Score (迭代 {args.metric_iteration} 次)...')
    discriminative_score = list()
    for i in range(args.metric_iteration):
        temp_disc = discriminative_score_metrics(ori_data, generated_data)
        discriminative_score.append(temp_disc)
        print(f'    迭代 {i+1}/{args.metric_iteration}: {temp_disc:.4f}')

    metric_results['discriminative'] = np.mean(discriminative_score)
    print(f'  ✅ 平均 Discriminative Score: {metric_results["discriminative"]:.4f}')

    # 2. Predictive score
    print(f'\n  计算 Predictive Score (迭代 {args.metric_iteration} 次)...')
    predictive_score = list()
    for tt in range(args.metric_iteration):
        temp_pred = predictive_score_metrics(ori_data, generated_data)
        predictive_score.append(temp_pred)
        print(f'    迭代 {tt+1}/{args.metric_iteration}: {temp_pred:.4f}')

    metric_results['predictive'] = np.mean(predictive_score)
    print(f'  ✅ 平均 Predictive Score: {metric_results["predictive"]:.4f}')

    # ==================== 4. Save everything ====================
    saved_files = save_model_and_data(
        model_dict=model_dict,
        generated_data=generated_data,
        ori_data=ori_data,
        args=args,
        metric_results=metric_results,
        save_dir=args.save_dir
    )

    # ==================== 5. Visualization ====================
    print('\n📈 生成可视化图表...')
    try:
        visualization(ori_data, generated_data, 'pca')
        print('  ✅ PCA 可视化完成')
    except Exception as e:
        print(f'  ⚠️  PCA 可视化失败: {str(e)}')

    try:
        visualization(ori_data, generated_data, 'tsne')
        print('  ✅ t-SNE 可视化完成')
    except Exception as e:
        print(f'  ⚠️  t-SNE 可视化失败: {str(e)}')

    # ==================== 6. Print summary ====================
    print_summary(saved_files, metric_results, args)

    return ori_data, generated_data, metric_results, saved_files


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser(
        description='TimeGAN - Time-series Generative Adversarial Networks (PyTorch Implementation)'
    )

    # Data parameters
    parser.add_argument(
        '--data_name',
        choices=['sine', 'stock', 'energy','wave'],
        default='wave',
        type=str,
        help='Dataset name: sine (synthetic), stock, or energy')

    parser.add_argument(
        '--seq_len',
        default=24,
        type=int,
        help='Sequence length of time-series data')

    # Network parameters
    parser.add_argument(
        '--module',
        choices=['gru', 'lstm', 'lstmLN'],
        default='gru',
        type=str,
        help='RNN module type: gru, lstm, or lstmLN')

    parser.add_argument(
        '--hidden_dim',
        default=24,
        type=int,
        help='Hidden state dimensions (should be optimized for different datasets)')

    parser.add_argument(
        '--num_layer',
        default=3,
        type=int,
        help='Number of RNN layers (should be optimized)')

    # Training parameters
    parser.add_argument(
        '--iteration',
        default=50000,
        type=int,
        help='Number of training iterations (should be optimized)')

    parser.add_argument(
        '--batch_size',
        default=128,
        type=int,
        help='Number of samples in mini-batch (should be optimized)')

    # Evaluation parameters
    parser.add_argument(
        '--metric_iteration',
        default=10,
        type=int,
        help='Number of iterations for metric computation')

    # Save parameters
    parser.add_argument(
        '--save_dir',
        default='./saved_models',
        type=str,
        help='Directory to save trained models and generated data')

    args = parser.parse_args()

    # Calls main function
    ori_data, generated_data, metrics, saved_files = main(args)

    print('\n' + '=' * 70)
    print('🎉 实验完成!')
    print('=' * 70)