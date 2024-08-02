import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
    def __init__(self, max_filter_num, img_size, img_channel):
        super(Generator, self).__init__()

        self.init_size = img_size // 16

        self.z_dim = max_filter_num * self.init_size ** 2
        self.l1 = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            # nn.ReLU(),
            # nn.Linear(feature_dim, feature_dim),
        )

        self.conv_blocks = nn.Sequential(

            nn.Conv2d(max_filter_num, max_filter_num, 3, stride=1, padding=1),
            nn.BatchNorm2d(max_filter_num),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(max_filter_num, max_filter_num // 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(max_filter_num // 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(max_filter_num // 2,  max_filter_num // 4, 3, stride=1, padding=1),
            nn.BatchNorm2d( max_filter_num // 4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(max_filter_num // 4, max_filter_num // 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(max_filter_num // 8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(max_filter_num // 8, max_filter_num // 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(max_filter_num // 8),
            nn.ReLU(),
            nn.Conv2d(max_filter_num // 8, img_channel, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class LargeGenerator(nn.Module):
    def __init__(self, max_filter_num, img_size, img_channel):
        super().__init__()

        self.init_size = img_size // 32

        self.z_dim = 4096
        self.z_dim_out = max_filter_num * self.init_size ** 2
        self.l1 = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim_out),
            # nn.ReLU(),
            # nn.Linear(feature_dim, feature_dim),
        )

        self.conv_blocks = nn.Sequential(

            nn.Conv2d(max_filter_num, max_filter_num, 3, stride=1, padding=1),
            nn.BatchNorm2d(max_filter_num),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(max_filter_num, max_filter_num // 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(max_filter_num // 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(max_filter_num // 2,  max_filter_num // 4, 3, stride=1, padding=1),
            nn.BatchNorm2d( max_filter_num // 4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(max_filter_num // 4, max_filter_num // 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(max_filter_num // 8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(max_filter_num // 8, max_filter_num // 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(max_filter_num // 16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(max_filter_num // 16, max_filter_num // 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(max_filter_num // 16),
            nn.ReLU(),
            nn.Conv2d(max_filter_num // 16, img_channel, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


