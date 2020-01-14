# -*- coding: utf-8 -*-

import os
import numpy as np

class Compare_Minimac_Dataset(object):
    def __init__(self,args):
        self.args = args

    def _load_data(self):
        if os.path.exists('../tmp/compare.npy'):
            data = np.load('../tmp/compare.npy')
        else:
            data = np.loadtxt(self.args.compare_datafile)
            data = data.T
        print('全部数据纬度 : {}'.format(data.shape))
        rate = np.sum(data, axis=0) / data.shape[0]
        np.save('../statstics/compare_rate.npy', rate)
        train_data, test_data = data[:int(data.shape[0]*0.8),...],data[int(data.shape[0] * 0.8):, ...]
        np.save('../tmp/compare_testdata.npy', test_data)
        rare_indexes = np.where(np.logical_or(rate <=0.05, rate>=0.95))[0]
        print("数据丢失率 ： {}".format(len(rare_indexes) / train_data.shape[1]))
        return train_data, test_data, rare_indexes, rate

    def _load_mask(self):
        if os.path.exists(self.args.compare_maskfile):
            mask = np.loadtxt(self.args.compare_maskfile)
            return mask

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
        np.random.seed(self.args.seed)
        return np.random.uniform(0.0,0.1,(m, n))

    def sample_idx(self, m, sample_num):
        """
        随机抽样
        """
        A = np.random.permutation(m)
        idx = A[:sample_num]
        return idx


if __name__ == "__main__":
    pass
