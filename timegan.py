"""Time-series Generative Adversarial Networks (TimeGAN) Codebase - PyTorch Implementation

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: 2025-10-17
Converted to PyTorch by: GitHub Copilot

-----------------------------

timegan.py (PyTorch Version)

Note: Use original data as training set to generate synthetic data (time-series)
"""

# Necessary Packages
import torch
import torch.nn as nn
import numpy as np
from utils import extract_time, batch_generator, random_generator


class Embedder(nn.Module):
    """Embedding network between original feature space to latent space."""

    def __init__(self, input_dim, hidden_dim, num_layers, module_name='gru'):
        super(Embedder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # RNN layer
        if module_name == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        elif module_name == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        elif module_name == 'lstmLN':
            # PyTorch doesn't have built-in LayerNorm LSTM, use regular LSTM
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Apply fully connected layer
        h = self.sigmoid(self.fc(output))
        return h


class Recovery(nn.Module):
    """Recovery network from latent space to original space."""

    def __init__(self, hidden_dim, output_dim, num_layers, module_name='gru'):
        super(Recovery, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # RNN layer
        if module_name == 'gru':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        elif module_name == 'lstm':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        elif module_name == 'lstmLN':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h, lengths):
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Apply fully connected layer
        x_tilde = self.sigmoid(self.fc(output))
        return x_tilde


class Generator(nn.Module):
    """Generator function: Generate time-series data in latent space."""

    def __init__(self, z_dim, hidden_dim, num_layers, module_name='gru'):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # RNN layer
        if module_name == 'gru':
            self.rnn = nn.GRU(z_dim, hidden_dim, num_layers, batch_first=True)
        elif module_name == 'lstm':
            self.rnn = nn.LSTM(z_dim, hidden_dim, num_layers, batch_first=True)
        elif module_name == 'lstmLN':
            self.rnn = nn.LSTM(z_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, lengths):
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Apply fully connected layer
        e = self.sigmoid(self.fc(output))
        return e


class Supervisor(nn.Module):
    """Generate next sequence using the previous sequence."""

    def __init__(self, hidden_dim, num_layers, module_name='gru'):
        super(Supervisor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers - 1  # One less layer as per original implementation

        # RNN layer
        if module_name == 'gru':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, self.num_layers, batch_first=True)
        elif module_name == 'lstm':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, self.num_layers, batch_first=True)
        elif module_name == 'lstmLN':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, self.num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h, lengths):
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Apply fully connected layer
        s = self.sigmoid(self.fc(output))
        return s


class Discriminator(nn.Module):
    """Discriminate the original and synthetic time-series data."""

    def __init__(self, hidden_dim, num_layers, module_name='gru'):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # RNN layer
        if module_name == 'gru':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        elif module_name == 'lstm':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        elif module_name == 'lstmLN':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer (no activation for logits)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h, lengths):
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Apply fully connected layer
        y_hat = self.fc(output)
        return y_hat


def timegan(ori_data, parameters):
    """TimeGAN function (PyTorch Implementation).

    Use original data as training set to generate synthetic data (time-series)

    Args:
        - ori_data: original time-series data
        - parameters: TimeGAN network parameters

    Returns:
        - generated_data: generated time-series data
    """

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)

    def MinMaxScaler(data):
        """Min-Max Normalizer.

        Args:
            - data: raw data

        Returns:
            - norm_data: normalized data
            - min_val: minimum values (for renormalization)
            - max_val: maximum values (for renormalization)
        """
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val

        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)

        return norm_data, min_val, max_val

    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    # Network Parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    z_dim = dim
    gamma = 1

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize networks
    embedder = Embedder(dim, hidden_dim, num_layers, module_name).to(device)
    recovery = Recovery(hidden_dim, dim, num_layers, module_name).to(device)
    generator = Generator(z_dim, hidden_dim, num_layers, module_name).to(device)
    supervisor = Supervisor(hidden_dim, num_layers, module_name).to(device)
    discriminator = Discriminator(hidden_dim, num_layers, module_name).to(device)

    # Optimizers
    e_optimizer = torch.optim.Adam(list(embedder.parameters()) + list(recovery.parameters()))
    d_optimizer = torch.optim.Adam(discriminator.parameters())
    g_optimizer = torch.optim.Adam(list(generator.parameters()) + list(supervisor.parameters()))
    gs_optimizer = torch.optim.Adam(list(generator.parameters()) + list(supervisor.parameters()))

    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    # ==================== 1. Embedding network training ====================
    print('Start Embedding Network Training')

    for itt in range(iterations):
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)

        # Convert to tensors
        X_mb = torch.FloatTensor(np.array(X_mb)).to(device)
        T_mb = torch.LongTensor(T_mb).cpu()  # Keep on CPU for pack_padded_sequence

        # Forward pass
        H = embedder(X_mb, T_mb)
        X_tilde = recovery(H, T_mb)

        # Compute loss
        E_loss_T0 = mse_loss(X_mb, X_tilde)
        E_loss0 = 10 * torch.sqrt(E_loss_T0)

        # Backward pass and optimize
        e_optimizer.zero_grad()
        E_loss0.backward()
        e_optimizer.step()

        # Checkpoint
        if itt % 1000 == 0:
            print(f'step: {itt}/{iterations}, e_loss: {np.round(np.sqrt(E_loss_T0.item()), 4)}')

    print('Finish Embedding Network Training')

    # ==================== 2. Training with supervised loss ====================
    print('Start Training with Supervised Loss Only')

    for itt in range(iterations):
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

        # Convert to tensors
        X_mb = torch.FloatTensor(np.array(X_mb)).to(device)
        Z_mb = torch.FloatTensor(np.array(Z_mb)).to(device)
        T_mb = torch.LongTensor(T_mb).cpu()

        # Forward pass
        H = embedder(X_mb, T_mb)
        E_hat = generator(Z_mb, T_mb)
        H_hat_supervise = supervisor(H, T_mb)

        # Supervised loss
        G_loss_S = mse_loss(H[:, 1:, :], H_hat_supervise[:, :-1, :])

        # Backward pass and optimize
        gs_optimizer.zero_grad()
        G_loss_S.backward()
        gs_optimizer.step()

        # Checkpoint
        if itt % 1000 == 0:
            print(f'step: {itt}/{iterations}, s_loss: {np.round(np.sqrt(G_loss_S.item()), 4)}')

    print('Finish Training with Supervised Loss Only')

    # ==================== 3. Joint Training ====================
    print('Start Joint Training')

    for itt in range(iterations):
        # Generator training (twice more than discriminator training)
        for kk in range(2):
            # Set mini-batch
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

            # Convert to tensors
            X_mb = torch.FloatTensor(np.array(X_mb)).to(device)
            Z_mb = torch.FloatTensor(np.array(Z_mb)).to(device)
            T_mb_cpu = torch.LongTensor(T_mb).cpu()

            # Forward pass - Embedder & Recovery
            H = embedder(X_mb, T_mb_cpu)
            X_tilde = recovery(H, T_mb_cpu)

            # Forward pass - Generator
            E_hat = generator(Z_mb, T_mb_cpu)
            H_hat = supervisor(E_hat, T_mb_cpu)
            H_hat_supervise = supervisor(H, T_mb_cpu)

            # Synthetic data
            X_hat = recovery(H_hat, T_mb_cpu)

            # Forward pass - Discriminator
            Y_fake = discriminator(H_hat, T_mb_cpu)
            Y_real = discriminator(H, T_mb_cpu)
            Y_fake_e = discriminator(E_hat, T_mb_cpu)

            # Generator loss
            # 1. Adversarial loss
            G_loss_U = bce_loss(Y_fake, torch.ones_like(Y_fake))
            G_loss_U_e = bce_loss(Y_fake_e, torch.ones_like(Y_fake_e))

            # 2. Supervised loss
            G_loss_S = mse_loss(H[:, 1:, :], H_hat_supervise[:, :-1, :])

            # 3. Two moments
            G_loss_V1 = torch.mean(torch.abs(
                torch.sqrt(X_hat.var(0, unbiased=False) + 1e-6) -
                torch.sqrt(X_mb.var(0, unbiased=False) + 1e-6)
            ))
            G_loss_V2 = torch.mean(torch.abs(X_hat.mean(0) - X_mb.mean(0)))
            G_loss_V = G_loss_V1 + G_loss_V2

            # 4. Total generator loss
            G_loss = G_loss_U + gamma * G_loss_U_e + 50 * torch.sqrt(G_loss_S) + 100 * G_loss_V

            # Backward pass - Generator
            g_optimizer.zero_grad()
            G_loss.backward()
            g_optimizer.step()

            # Embedder training
            H = embedder(X_mb, T_mb_cpu)
            X_tilde = recovery(H, T_mb_cpu)
            H_hat_supervise = supervisor(H, T_mb_cpu)

            E_loss_T0 = mse_loss(X_mb, X_tilde)
            E_loss = 10 * torch.sqrt(E_loss_T0) + 0.1 * mse_loss(H[:, 1:, :], H_hat_supervise[:, :-1, :])

            e_optimizer.zero_grad()
            E_loss.backward()
            e_optimizer.step()

        # Discriminator training
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

        # Convert to tensors
        X_mb = torch.FloatTensor(np.array(X_mb)).to(device)
        Z_mb = torch.FloatTensor(np.array(Z_mb)).to(device)
        T_mb_cpu = torch.LongTensor(T_mb).cpu()

        # Forward pass
        H = embedder(X_mb, T_mb_cpu)
        E_hat = generator(Z_mb, T_mb_cpu)
        H_hat = supervisor(E_hat, T_mb_cpu)

        Y_real = discriminator(H, T_mb_cpu)
        Y_fake = discriminator(H_hat, T_mb_cpu)
        Y_fake_e = discriminator(E_hat, T_mb_cpu)

        # Discriminator loss
        D_loss_real = bce_loss(Y_real, torch.ones_like(Y_real))
        D_loss_fake = bce_loss(Y_fake, torch.zeros_like(Y_fake))
        D_loss_fake_e = bce_loss(Y_fake_e, torch.zeros_like(Y_fake_e))
        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        # Train discriminator only when it doesn't work well
        if D_loss.item() > 0.15:
            d_optimizer.zero_grad()
            D_loss.backward()
            d_optimizer.step()

        # Print checkpoints
        if itt % 1000 == 0:
            print(f'step: {itt}/{iterations}, ' +
                  f'd_loss: {np.round(D_loss.item(), 4)}, ' +
                  f'g_loss_u: {np.round(G_loss_U.item(), 4)}, ' +
                  f'g_loss_s: {np.round(np.sqrt(G_loss_S.item()), 4)}, ' +
                  f'g_loss_v: {np.round(G_loss_V.item(), 4)}, ' +
                  f'e_loss_t0: {np.round(np.sqrt(E_loss_T0.item()), 4)}')

    print('Finish Joint Training')

    # ==================== Synthetic data generation ====================
    embedder.eval()
    generator.eval()
    supervisor.eval()
    recovery.eval()

    with torch.no_grad():
        Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
        Z_mb = torch.FloatTensor(np.array(Z_mb)).to(device)
        ori_time_tensor = torch.LongTensor(ori_time).cpu()

        E_hat = generator(Z_mb, ori_time_tensor)
        H_hat = supervisor(E_hat, ori_time_tensor)
        generated_data_curr = recovery(H_hat, ori_time_tensor)
        generated_data_curr = generated_data_curr.cpu().numpy()

    generated_data = list()

    for i in range(no):
        temp = generated_data_curr[i, :ori_time[i], :]
        generated_data.append(temp)

    # Renormalization
    generated_data = np.array(generated_data)
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val

    # ==================== 保存模型 ====================
    # 创建模型字典
    model_dict = {
        'embedder': embedder.state_dict(),
        'recovery': recovery.state_dict(),
        'generator': generator.state_dict(),
        'supervisor': supervisor.state_dict(),
        'discriminator': discriminator.state_dict(),
        'parameters': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'module_name': module_name,
            'z_dim': z_dim,
            'dim': dim,
            'max_seq_len': max_seq_len
        },
        'normalization': {
            'min_val': min_val,
            'max_val': max_val
        }
    }

    print('\n✅ Model dictionary created successfully!')

    return generated_data, model_dict