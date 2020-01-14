# -*- coding: utf-8 -*-

################################################################
# author : zhaojifan
# date : 2019/09/15
# email : zhaojifan@23mofang.com
# @copyright 23mofang CO.
###############################################################

import torch
import torch.nn as nn
import numpy as np
import random
import os
import time
import torch.nn.functional as F
from torch.autograd import Variable


import warnings
warnings.filterwarnings("ignore")

# 初始化权重
def _weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

##################################################################################################
# Focal Loss
##################################################################################################
class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        loss = - self.alpha*(1 - pt) ** self.gamma * target * torch.log(pt) - \
            (1-self.alpha)*pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

####################################################################################################
# full + conv
####################################################################################################
class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.conv_sequential = self._build_conv_submodule()
        self.linear_sequential = self._build_linear_submodule()
        self.final_fc = torch.nn.Linear(args.dim, args.dim)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh =  torch.nn.Tanh()
        self.apply(_weight_init)
        self.w1 = nn.Parameter(torch.FloatTensor(args.dim).uniform_(0, 1))


    def _build_conv_submodule(self):
        if self.args.conv_channel_list[-1] != 1:
            self.args.conv_channel_list.append(1)
        conv_sequential = nn.Sequential()

        for i in range(len(self.args.conv_channel_list)-1):
            conv_sequential.add_module('conv_level_{}'.format(i), \
                            nn.Conv1d(self.args.conv_channel_list[i], self.args.conv_channel_list[i+1], \
                            kernel_size=self.args.kernel_size[i], padding=self.args.kernel_size[i]//2))
            conv_sequential.add_module('bn_level_{}'.format(i),\
                        nn.BatchNorm1d(self.args.conv_channel_list[i+1]))
            conv_sequential.add_module('activation_{}'.format(i), nn.LeakyReLU(0.2, inplace=True))
        return conv_sequential

    def _build_linear_submodule(self):
        if self.args.linear_size[-1] != self.args.dim:
            self.args.linear_size.append(self.args.dim)
        linear_sequential = nn.Sequential()

        for i in range(len(self.args.linear_size)):
            if not i:
                linear_sequential.add_module('linear_level_{}'.format(i), \
                                    nn.Linear(self.args.dim*2, self.args.linear_size[i]))
            else:
                linear_sequential.add_module('linear_level_{}'.format(i), \
                                    nn.Linear(self.args.linear_size[i-1], self.args.linear_size[i]))
            linear_sequential.add_module('activation_{}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            if self.args.drop_rate:
                linear_sequential.add_module('dropout_{}'.format(i), nn.Dropout(self.args.drop_rate))
        return linear_sequential

    def _conv_forward(self, x, z, m):
        inp = m * x + (1 - m) * z
        inp, m = torch.unsqueeze(inp, 1), torch.unsqueeze(m, 1)
        inp = torch.cat((inp, m), 1)
        out = self.conv_sequential(inp)
        out = out.view(out.size()[0], -1)
        return out

    def _linear_forward(self, x, z, m):
        inp = m * x + (1-m) * z
        inp = torch.cat((inp, m), dim=1)
        out = self.linear_sequential(inp)
        return out

    def forward(self, x, z, m, rate):
        if self.args.gconnection == 'linear':
            out = self._linear_forward(x, z, m)
            return self.sigmoid(out)
        elif self.args.gconnection == 'conv':
            out =  self._conv_forward(x, z, m)
            return self.sigmoid(out)
        elif self.args.gconnection == 'linear-conv':
            rate = torch.from_numpy(np.repeat(rate[np.newaxis,...], x.size()[0], axis=0)).float()
            conv_out = self._conv_forward(x, z, m)
            linear_out = self._linear_forward(x, z, m)
            final_inp = (conv_out + linear_out) * self.w1 + rate * (1 - self.w1)
            final_out = self.tanh(self.final_fc(final_inp))
            return final_out
        else:
            raise NotImplementedError('one of [linear | conv | linear-conv]')


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        if self.args.dconnection == 'linear':
            self.linear_sequential = self._build_linear_submodule()
        elif self.args.dconnection == 'conv':
            self.conv_sequential = self._build_conv_submodule()
        elif self.args.dconnection == 'linear-conv':
            self.conv_sequential = self._build_conv_submodule()
            self.linear_sequential = self._build_linear_submodule()
        self.final_fc = torch.nn.Linear(args.dim, args.dim)
        self.apply(_weight_init)

    def _build_conv_submodule(self):
        if self.args.conv_channel_list[-1] != 1:
            self.args.conv_channel_list.append(1)
        conv_sequential = nn.Sequential()

        for i in range(len(self.args.conv_channel_list)-1):
            conv_sequential.add_module('conv_level_{}'.format(i), \
                            nn.Conv1d(self.args.conv_channel_list[i], self.args.conv_channel_list[i+1], \
                            kernel_size=self.args.kernel_size[i], padding=self.args.kernel_size[i]//2))
            conv_sequential.add_module('bn_level_{}'.format(i), nn.BatchNorm1d(self.args.conv_channel_list[i+1]))
            conv_sequential.add_module('activation_{}'.format(i), nn.ReLU())
        return conv_sequential

    def _build_linear_submodule(self):
        if self.args.linear_size[0] != self.args.dim:
            self.args.linear_size.append(self.args.dim)
        linear_sequential = nn.Sequential()

        for i in range(len(self.args.linear_size)):
            if not i:
                linear_sequential.add_module('linear_level_{}'.format(i), \
                                    nn.Linear(self.args.dim*2, self.args.linear_size[i]))
            else:
                linear_sequential.add_module('linear_level_{}'.format(i), \
                                    nn.Linear(self.args.linear_size[i-1], self.args.linear_size[i]))
            linear_sequential.add_module('activation_{}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            if self.args.drop_rate:
                linear_sequential.add_module('dropout_{}'.format(i), nn.Dropout(self.args.drop_rate))
        return linear_sequential

    def _conv_forward(self, x, m, g, h):
        inp = m * x + (1 - m) * g
        inp, h = torch.unsqueeze(inp, 1), torch.unsqueeze(h, 1)
        inp = torch.cat((inp, h), 1)
        out = self.conv_sequential(inp)
        out = out.view(out.size()[0], -1)
        return out

    def _linear_forward(self, x, m, g, h):
        inp = m * x + (1-m) * g
        inp = torch.cat((inp, h), dim=1)
        out = self.linear_sequential(inp)
        return out

    def forward(self, x, m, g, h):
        if self.args.dconnection == 'linear':
            out = self._linear_forward(x, m, g, h)
            return out
        elif self.args.dconnection == 'conv':
            out =  self._conv_forward(x, m, g, h)
            return out
        elif self.args.dconnection == 'linear-conv':
            conv_out = self._conv_forward(x, m, g, h)
            linear_out = self._linear_forward(x, m, g, h)
            final_inp = conv_out + linear_out
            final_out = self.final_fc(final_inp)
            return final_out
        else:
            raise NotImplementedError('one of [linear | conv | linear-conv]')

if __name__ == "__main__":
    from args import args
    g = Generator(args)
    print(g)
