import copy
import math

import torch
import torch.nn.functional as F
from mmdet.models import BACKBONES
from torch import nn as nn
import numpy as np
import torch

from torch.nn import functional as F

from mmdet3d.models.losses import chamfer_distance

from mmdet.core import multi_apply
import copy
import warnings
from abc import ABCMeta
from collections import defaultdict
from logging import FileHandler
from mmcv.runner.dist_utils import master_only
from mmcv.utils.logging import get_logger, logger_initialized, print_log


@BACKBONES.register_module()
class pointnet(nn.Module):
    def __init__(self, input_channel, init_cfg=None):
        super(pointnet, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x_trans = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x_trans)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x


@BACKBONES.register_module()
class inducer_attention(nn.Module):
    def __init__(self, input_channel, ratio=8):
        super(inducer_attention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channel, out_channels=input_channel // ratio, kernel_size=1,
                               bias=False)
        self.attention_bn1 = nn.BatchNorm1d(input_channel // ratio)

        self.conv2 = nn.Conv1d(in_channels=input_channel, out_channels=input_channel // ratio, kernel_size=1,
                               bias=False)
        self.attention_bn2 = nn.BatchNorm1d(input_channel // ratio)

        self.conv3 = nn.Conv1d(
            in_channels=input_channel, out_channels=input_channel, kernel_size=1, bias=False)
        self.attention_bn3 = nn.BatchNorm1d(input_channel)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        b, c, n = q.shape
        a = F.relu(self.attention_bn1(self.conv1(k))).permute(0, 2, 1)
        b = F.relu(self.attention_bn2(self.conv2(q)))  # b, c/ratio, n
        s = self.softmax(torch.bmm(a, b)) / math.sqrt(c)  # b,n,n
        d = F.relu(self.attention_bn3(self.conv3(v)))  # b,c,n
        out = q + torch.bmm(d, s.permute(0, 2, 1))
        return out
