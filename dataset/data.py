# -*- coding: utf-8 -*-

################################################################
# author : zhaojifan
# date : 2019/09/15
# email : zhaojifan@23mofang.com
# @copyright 23mofang CO.
###############################################################

import os
import random
import numpy as np
from sklearn.externals import joblib

class RandomDataset(object):
    def __init__(self, args):
        self.args = args

    def load_snp(self):
        if os.path.exists(os.path.join(self.args.datapath, 'impute_data.npy')):
            data = np.load(os.path.join(self.args.datapath, 'impute_data.npy'))
        else:
            data = joblib.load(os.path.join(self.args.datapath,'data.pkl'))
            np.save(os.path.join(self.args.datapath, 'impute_data.npy'), data)
        data = data.T
        random.seed(self.args.seed)
        if self.args.dim != data.shape[1]:
            start = random.randint(0,data.shape[1]-self.args.dim)

        else:
            start = 0
        data = data[:, start : start + self.args.dim]
        rate = np.sum(data, axis=0) / data.shape[0]
        np.save('rate.npy', rate)
        rare_indexes = np.where(np.logical_or(rate <=0.05, rate>=0.95))[0]
        print('rate : {}'.format(len(rare_indexes)/self.args.dim))
        random.seed()
        permutation = np.random.permutation(data.shape[0])
        data = data[permutation, ...]
        trainset, testset = data[:int(self.args.split_rate * data.shape[0]), ...], \
                            data[int(self.args.split_rate * data.shape[0]):, ...]
        np.save('test.npy',testset)
        return trainset, testset, start, rare_indexes, rate

    def _load_data(self):
        print('加载对比数据')
        if os.path.exists('tmp/compare.npy'):
            data = np.load('tmp/compare.npy')
        else:
            data = np.loadtxt(self.args.compare_datafile)
            data = data.T
            np.save('tmp/compare.npy', data)
        rate = np.sum(data, axis=0) / data.shape[0]
        np.save('statstics/compare_rate.npy', rate)
        train_data, test_data = data[:int(data.shape[0]*0.8),...],data[int(data.shape[0] * 0.8):, ...]
        print('数据总共{}条 --> 训练数据{}条 验证数据{}条'.format(data.shape, train_data.shape, test_data.shape))
        np.save('tmp/compare_testdata.npy', test_data)
        rare_indexes = np.where(np.logical_or(rate <=0.05, rate>=0.95))[0]
        return train_data, test_data,0, rare_indexes, rate

    #随机产生MASK
    def sample_M(self, m, n, p):
        """
        :parameters
        (m, n) 输入的纬度，  m个samples， n是数据的特征纬度
        p 截止概率
        """
        A = np.random.uniform(0, 1, (m, n))
        B = A > p
        return 1.0 * B

    def sample_Z(self, m,n):
        """
        产生随机噪声
        """
        return np.random.uniform(0.0,1.0,(m, n))

    def sample_idx(self, m, sample_num):
        """
        随机抽样
        """
        A = np.random.permutation(m)
        idx = A[:sample_num]
        return idx
