# -*- coding: utf-8 -*-

################################################################
# author : zhaojifan
# date : 2019/10/8
# email : zhaojifan@23mofang.com
# @copyright 23mofang CO.
###############################################################

import sys
sys.path.append('../')

import os
import time
import torch
import random
from compare_dataset import Compare_Minimac_Dataset
from module.network import Generator, Discriminator, BCEFocalLoss
from utils import *

def run(args):
    dataset = Compare_Minimac_Dataset(args)
    trainset, testset,  rare_indexes, rate = dataset._load_data()
    print('训练数据纬度 ：{}   测试数据纬度：{}'.format(trainset.shape, testset.shape))
    print('数据加载完毕')
    train_mask, test_mask \
        = dataset.sample_M(trainset.shape[0], trainset.shape[1], args.miss_p), \
        dataset.sample_M(testset.shape[0], testset.shape[1], args.miss_p)

    print('数据处理完成')
    if args.save:
        now = int(time.time())
        timeStruct = time.localtime(now)
        save_dir = time.strftime("%Y%m%d-%H%M%S", timeStruct)
        save_path = os.path.join(args.save_path, "compare_"+save_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    netD = Discriminator(args)
    netG = Generator(args)

    optimD = torch.optim.Adam(netD.parameters(), lr=0.002)
    optimG = torch.optim.Adam(netG.parameters(), lr=0.002)
    scheduleD = torch.optim.lr_scheduler.StepLR(optimD, step_size=100, gamma=0.9)
    scheduleG = torch.optim.lr_scheduler.StepLR(optimG, step_size=100, gamma=0.9)
    #bce_loss = torch.nn.BCEWithLogitsLoss(reduction="elementwise_mean")
    bce_loss = BCEFocalLoss(alpha=0.25)
    mse_loss = torch.nn.MSELoss(reduction="elementwise_mean")
    print('初始化完成，开始训练.......')
    test_mb_idx = dataset.sample_idx(testset.shape[0], args.test_sampling_num)
    for epoch in range(args.num_epochs+1):
        #scheduleD.step()
        #scheduleG.step()
        netD.train()
        netG.train()
        with torch.set_grad_enabled(True):
            # 随机抽取训练样本
            for _ in range(5):
                mb_idx = dataset.sample_idx(trainset.shape[0], args.batch_size)
                X_mb = trainset[mb_idx,:]
                Z_mb = dataset.sample_Z(args.batch_size, args.dim)
                M_mb = dataset.sample_M(X_mb.shape[0], args.dim, args.miss_p)
                H_mb1 = dataset.sample_M(args.batch_size, args.dim, 1-args.hint_rate)
                H_mb = M_mb * H_mb1 + 0.5*(1-H_mb1)
                New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
                X_mb = torch.tensor(X_mb).float()
                New_X_mb = torch.tensor(New_X_mb).float()
                Z_mb = torch.tensor(Z_mb).float()
                M_mb = torch.tensor(M_mb).float()
                H_mb = torch.tensor(H_mb).float()
                # Train D
                G_sample = netG(X_mb, New_X_mb, M_mb, rate)
                D_prob = netD(X_mb, M_mb, G_sample, H_mb)
                D_loss = bce_loss(D_prob, M_mb)

                optimD.zero_grad()
                D_loss.backward()
                optimD.step()

            mb_idx = dataset.sample_idx(trainset.shape[0], args.batch_size)
            X_mb = trainset[mb_idx,:]
            #if random.uniform(0,1) >= 0.65:
            #    X_mb = random_flip(X_mb, 0.02)
            Z_mb = dataset.sample_Z(args.batch_size, args.dim)
            M_mb = train_mask[mb_idx,:]
            H_mb1 = dataset.sample_M(args.batch_size, args.dim, 1-args.hint_rate)
            H_mb = M_mb * H_mb1 + 0.5*(1-H_mb1)
            New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
            X_mb = torch.tensor(X_mb).float()
            New_X_mb = torch.tensor(New_X_mb).float()
            Z_mb = torch.tensor(Z_mb).float()
            M_mb = torch.tensor(M_mb).float()
            H_mb = torch.tensor(H_mb).float()
            # Train G
            G_sample = netG(X_mb, New_X_mb, M_mb, rate)
            D_prob = netD(X_mb, M_mb, G_sample, H_mb)
            D_prob.detach_()
            G_loss1 = ((1 - M_mb) * (torch.sigmoid(D_prob)+1e-8).log()).mean()/(1-M_mb).sum()
            # MSE Loss
            loss_rare = np.repeat(rate[np.newaxis,...], args.batch_size, axis=0)
            loss_rare = torch.FloatTensor(loss_rare)
            G_mse_loss = mse_loss(M_mb*X_mb, M_mb*G_sample) / M_mb.sum()
            G_loss = G_loss1 + args.alpha*G_mse_loss
            optimG.zero_grad()
            G_loss.backward()
            optimG.step()

        if epoch % args.print_freq == 0:
            print('[Train]Epoch: {:4d}   [Train]DLoss: {:.4f}    [Train]GLoss: {:.4f}'.format(epoch, D_loss, G_loss))
            netG.eval()
            netD.eval()
            with torch.set_grad_enabled(False):
                X_mb = testset[test_mb_idx,:]
                M_mb = test_mask[test_mb_idx,:]
                Z_mb = dataset.sample_Z(args.test_sampling_num, args.dim)
                New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
                X_mb = torch.tensor(X_mb).float()
                New_X_mb = torch.tensor(New_X_mb).float()
                Z_mb = torch.tensor(Z_mb).float()
                M_mb = torch.tensor(M_mb).float()
                samples1 = X_mb
                samples5 = M_mb * X_mb + (1-M_mb) * Z_mb
                samples2 = netG(X_mb, New_X_mb, M_mb, rate)
                samples2 = M_mb * X_mb + (1-M_mb) * samples2

                acc,f1 = pos_acc(samples2, samples1, M_mb, rare_indexes, 0.5)
                print('本次测试中一共含有 {} 位点， 未抽取的共 {} 位点'.format(M_mb.shape[1], np.sum(acc<0)))
                print('[Macro F1>=0.99]({:.4f})  [Macro F1>=0.96]({:.4f})  [Macro F1>=0.9]{:.4f}'.format(\
                        np.sum(f1>=0.99)/len(f1), np.sum(f1>=0.96)/len(f1), np.sum(f1>=0.9)/len(f1)))
                print('总体平均F1-score为：{:.4f}, 总体F1-score标准差：{:.4f}'.format(np.mean(f1), np.std(f1)))
                print('='*120)
                if args.save:
                    save_model(save_path, netG, 0, epoch,
                                [np.sum(f1>=0.99)/len(f1), np.sum(f1>=0.96)/len(f1), np.sum(f1>=0.9)/len(f1)])
                print('\n')


if __name__ == "__main__":
    from args import args
    print('\n')
    print('='*200)
    print('训练参数：', end='')
    print(args)
    run(args)
