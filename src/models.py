import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, p_drop=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        self.batchnorm2 = nn.BatchNorm1d(num_features=out_dim)
        self.dropout = nn.Dropout(p_drop)

        if in_dim != out_dim:
            self.residual = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, X):
        residual = self.residual(X)
        X = F.leaky_relu(self.batchnorm1(self.conv1(X)))
        X = self.dropout(X)
        X = self.batchnorm2(self.conv2(X))
        X += residual
        return F.leaky_relu(X)

class ImprovedConvClassifier(nn.Module):
    def __init__(self, num_classes, seq_len, in_channels, hid_dim=128, p_drop=0.3):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(in_channels, hid_dim, p_drop=p_drop),
            ResidualBlock(hid_dim, hid_dim, p_drop=p_drop),
            ResidualBlock(hid_dim, hid_dim * 2, p_drop=p_drop),
            ResidualBlock(hid_dim * 2, hid_dim * 2, p_drop=p_drop),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim * 2, num_classes),
        )

    def forward(self, X):
        X = self.blocks(X)
        return self.head(X)
