# -*- coding: utf-8 -*-

################################################################
# author : zhaojifan
# date : 2019/09/15
# email : zhaojifan@23mofang.com
# @copyright 23mofang CO.
###############################################################

import torch
import os
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score

def calc_accuracy(samplesA, samplesB, M, epoch):
    samplesC= samplesA.detach().numpy()
    samplesA = np.uint8(samplesA.detach().numpy() >= 0.5)
    samplesB = np.uint8(samplesB.detach().numpy())
    acc = []
    missing_all, filling_all = None, None
    for i in range(M.shape[0]):
        filling = np.uint8(samplesA[i][M[i]==0])
        missing = np.uint8(samplesB[i][M[i]==0])
        missing_all = missing if missing_all is None else np.hstack((missing_all, missing))
        filling_all = filling if filling_all is None else np.hstack((filling_all, filling))
        sample = samplesC[i][M[i]==0]
        ac = np.sum(missing == filling) / len(missing)
        acc.append(ac)
    print("混淆矩阵：")
    print(confusion_matrix(missing_all, filling_all))
    print('分类报告：')
    print(classification_report(missing_all, filling_all))
    return np.array(acc)

def greater_than_spect(arr, threshold):
    return np.sum(np.uint8([arr>=threshold])) / len(arr)

def random_flip(arr, threshold):
    p = np.random.uniform(0, 1, (arr.shape[0], arr.shape[1]))
    mask = p > threshold
    arr[~mask] = np.uint8(np.logical_not(arr[~mask]))
    return np.uint8(arr)

def save_model(save_path, model, start_pos, epoch, accuracy):
    torch.save({
        'model': model.state_dict(),
        'start_pos':start_pos
    }, os.path.join(save_path, '{}_{:.2f}_{:.2f}_{:.2f}.ckpt'.format(epoch, accuracy[0], accuracy[1], accuracy[2])))

def pos_acc(samplesA, samplesB, mask, rare_indexes, threshold):
    samplesA = np.uint8(samplesA.detach().numpy()>=threshold)
    samplesB = np.uint8(samplesB.detach().numpy() > 0.5)
    mask = np.uint8(mask.numpy())
    acc = []
    f1 = []
    pos_true, pos_pred = None, None
    for col in range(mask.shape[1]):
        if col in rare_indexes:
            continue
        col_mask = mask[:, col].flatten()

        if col_mask.astype(np.uint8).all():
            acc.append(-1)
            continue
        col_missing = samplesB[:, col].flatten()
        col_filling = samplesA[:, col].flatten()
        pos_true = col_missing[col_mask==0] if pos_true is None else np.hstack((pos_true, col_missing[col_mask==0]))
        pos_pred = col_filling[col_mask==0] if pos_pred is None else np.hstack((pos_pred, col_filling[col_mask==0]))
        col_acc = np.sum(col_missing[col_mask==0] == col_filling[col_mask==0])/len(col_missing[col_mask==0])
        acc.append(col_acc)
        f1.append(f1_score(col_missing[col_mask==0], col_filling[col_mask==0], average='macro'))
    acc = np.array(acc)
    f1 = np.array(f1)
    print('位点混淆矩阵：')
    print(confusion_matrix(pos_true, pos_pred))
    #auc = roc_auc_score(pos_true, pos_pred)
    return acc, f1


def error_analysis_by_pos(groundtruth, predict, start_pos, data_len, epoch):
    false_postive, false_negative = [], []
    for gt, pd in zip(groundtruth.T, predict.T):
        fp = fn = 0
        for g, p in zip(gt, pd):
            if g == 1 and p == 0:
                fn += 1
            if g == 0 and p == 1:
                fp += 1
        false_postive.append(fp)
        false_negative.append(fn)
    error_dict = {'pos_fp':np.array(false_postive), 'pos_fn' : np.array(false_negative), 'start' : start_pos}
    with open('error/epoch_%d_error_record_%d_%d.pk'%(epoch,  start_pos, start_pos+data_len), 'wb') as fr:
        pickle.dump(error_dict, fr)
