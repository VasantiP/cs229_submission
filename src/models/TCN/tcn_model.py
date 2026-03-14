"""
Temporal Convolutional Network (TCN) for binary classification.

Standalone model module — imported by training scripts.

Architecture follows Bai et al. (2018) with improvements:
  - Kaiming/He initialization (principled for ReLU activations)
  - Optional BatchNorm after convolutions (helps with small datasets)
  - Optional weight normalization
  - Configurable pooling strategy (mean, max, or attention)

Usage:
    from tcn_model import TCN

    model = TCN(
        num_inputs=2,
        num_channels=[32, 32, 32, 32],
        kernel_size=3,
        dropout=0.2,
        use_batch_norm=True,
        pooling='attention',
    )
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# ══════════════════════════════════════════════════════════════════════════════
# Building Blocks
# ══════════════════════════════════════════════════════════════════════════════

class Chomp1d(nn.Module):
    """Remove right-side padding to enforce causal convolution."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Single TCN residual block: two dilated causal convolutions with
    optional batch normalization and residual connection.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, dropout=0.2, use_batch_norm=False, use_weight_norm=False):
        super().__init__()

        # ── First conv layer ──────────────────────────────────────────────
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs) if use_batch_norm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # ── Second conv layer ─────────────────────────────────────────────
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs) if use_batch_norm else nn.Identity()
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2,
        )

        # ── Residual connection ───────────────────────────────────────────
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

        # ── Optional weight normalization ─────────────────────────────────
        if use_weight_norm:
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)
            if self.downsample is not None:
                self.downsample = weight_norm(self.downsample)

        # ── Initialize weights ────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Kaiming/He initialization — principled for ReLU activations."""
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# ══════════════════════════════════════════════════════════════════════════════
# Pooling Strategies
# ══════════════════════════════════════════════════════════════════════════════

class AttentionPooling(nn.Module):
    """Learned attention-weighted pooling over the time dimension."""

    def __init__(self, n_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_channels, n_channels // 2),
            nn.Tanh(),
            nn.Linear(n_channels // 2, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        Returns:
            pooled: (batch, channels)
        """
        # x -> (batch, time, channels) for attention computation
        x_t = x.permute(0, 2, 1)
        weights = self.attention(x_t)          # (batch, time, 1)
        weights = torch.softmax(weights, dim=1)
        pooled = (x_t * weights).sum(dim=1)    # (batch, channels)
        return pooled


# ══════════════════════════════════════════════════════════════════════════════
# Full TCN Model
# ══════════════════════════════════════════════════════════════════════════════

class TCN(nn.Module):
    """
    Temporal Convolutional Network for binary classification.

    Improvements over vanilla TCN:
        - Kaiming initialization (better gradient flow with ReLU)
        - Optional BatchNorm (stabilizes training on small datasets)
        - Optional weight normalization
        - Configurable pooling: 'mean', 'max', or 'attention'

    Receptive field:  R = 1 + 2 * (kernel_size - 1) * (2^num_levels - 1)
        Example: 4 layers, kernel_size=3  →  R = 63 timesteps
        Example: 6 layers, kernel_size=3  →  R = 255 timesteps
    """

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2,
                 use_batch_norm=True, use_weight_norm=False, pooling='mean'):
        """
        Args:
            num_inputs:       Number of input features per timestep
            num_channels:     List of channel widths per TCN layer, e.g. [32, 32, 32, 32]
            kernel_size:      Convolution kernel size (default 3)
            dropout:          Dropout rate (default 0.2)
            use_batch_norm:   Add BatchNorm after each conv (default True)
            use_weight_norm:  Apply weight normalization to convs (default False)
            pooling:          'mean', 'max', or 'attention' (default 'mean')
        """
        super().__init__()

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size

            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size, padding=padding,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                use_weight_norm=use_weight_norm,
            ))

        self.network = nn.Sequential(*layers)

        # ── Pooling ───────────────────────────────────────────────────────
        self.pooling_type = pooling
        if pooling == 'attention':
            self.pool = AttentionPooling(num_channels[-1])
        elif pooling == 'max':
            self.pool = None  # handled in forward
        elif pooling == 'mean':
            self.pool = None  # handled in forward
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")

        # ── Classification head ───────────────────────────────────────────
        self.fc = nn.Linear(num_channels[-1], 1)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    @property
    def receptive_field(self):
        """Compute the receptive field size in timesteps."""
        num_levels = len(self.network)
        # Extract kernel_size from first block
        k = self.network[0].conv1.kernel_size[0]
        return 1 + 2 * (k - 1) * (2 ** num_levels - 1)

    def forward(self, x):
        """
        Args:
            x: (batch, features, time)
        Returns:
            predictions: (batch,) with values in [0, 1]
        """
        y = self.network(x)  # (batch, channels, time)

        # Pool over time
        if self.pooling_type == 'attention':
            y = self.pool(y)                    # (batch, channels)
        elif self.pooling_type == 'max':
            y = y.max(dim=2).values             # (batch, channels)
        else:  # mean
            y = y.mean(dim=2)                   # (batch, channels)

        y = self.fc(y)                          # (batch, 1)
        y = torch.sigmoid(y).squeeze(-1)        # (batch,)
        return y


# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def count_parameters(model):
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    """Print a compact model summary."""
    print(f"TCN Summary:")
    print(f"  Receptive field:   {model.receptive_field} timesteps")
    print(f"  Pooling:           {model.pooling_type}")
    print(f"  Parameters:        {count_parameters(model):,}")
    print(f"  Architecture:      {[b.conv1.out_channels for b in model.network]}")