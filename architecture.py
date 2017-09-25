import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

class Baseline(nn.Module):
    def __init__(self, nTokens, nClasses):
        super(Baseline, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(10, 20, kernel_size=3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(20, nTokens, kernel_size=3, padding=1),
            nn.BatchNorm2d(nTokens),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(nTokens, nClasses)
    def forward(self, x):
        x = self.features(x)
        x = torch.squeeze(F.avg_pool2d(x, 6))
        x = self.fc(x)
        return F.log_softmax(x)
