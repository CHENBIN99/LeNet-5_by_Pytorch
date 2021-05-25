# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 第一层卷积
        self.conv_1 = nn.Conv2d(3, 6, (5, 5), stride=1, padding=2)
        # 第二层池化
        self.pool_1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2, padding=0)
        # 第三层卷积
        self.conv_2 = nn.Conv2d(6, 16, (5, 5), stride=1, padding=2)
        # 第四层池化
        self.pool_2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2, padding=0)
        # 第七层全连接
        self.linear_1 = nn.Linear(16 * 8 * 8, 120)
        # 第八层全连接
        self.linear_2 = nn.Linear(120, 84)
        # 第九层全连接
        self.linear_3 = nn.Linear(84, 10)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_c1 = self.relu(self.conv_1(x))
        x_p1 = self.pool_1(x_c1)
        x_c2 = self.relu(self.conv_2(x_p1))
        x_p2 = self.pool_2(x_c2)
        x_c3 = x_p2.view(x_p2.size(0), -1)
        x_l1 = self.relu(self.linear_1(x_c3))
        x_l2 = self.relu(self.linear_2(x_l1))
        x_l3 = self.linear_3(x_l2)
        return x_l3
