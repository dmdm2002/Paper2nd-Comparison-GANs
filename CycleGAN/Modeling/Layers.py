import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import matplotlib as plt
import math
import numpy as np


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class ResBlock(nn.Module):
    def __init__(self, f):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(f, f, 3, 1, 1),
            nn.InstanceNorm2d(f), nn.ReLU(),
            nn.Conv2d(f, f, 3, 1, 1),
        )
        self.norm = nn.InstanceNorm2d(f)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)+x))