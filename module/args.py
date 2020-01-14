# -*- coding: utf-8 -*-

################################################################
# author : zhaojifan
# date : 2019/09/15
# email : zhaojifan@23mofang.com
# @copyright 23mofang CO.
###############################################################
import argparse

args_parser = argparse.ArgumentParser(description='训练配置参数')
args_parser.add_argument('--dim', type=int, default=500, help='训练数据使用的window大小')
args_parser.add_argument('--miss_p', type=float, default=0.05, help='数据丢失率')
args_parser.add_argument('--batch_size', type=int, default=2, help='batch大小')
args_parser.add_argument('--hint_rate', type=float, default=0.9, help='避免模型坍塌的随机噪声率')
args_parser.add_argument('--alpha', type=float, default=9.9, help='损失函数的权重')
args_parser.add_argument('--split_rate', type=float, default=0.7, help='数据划分比例')
args_parser.add_argument('--gconnection', type=str, default='linear-conv', help='生成器内部连接方式')
args_parser.add_argument('--dconnection', type=str, default='linear', help='判别起内部连接方式')
args_parser.add_argument('--num_epochs', type=int, default=10000, help='由于每次随抽样训练，总的训练epoch应该大一些')
args_parser.add_argument('--save', type=bool, default=False, help='是否保存模型')
args_parser.add_argument('--save_path', type=str, default='/home/zhaojifan/impute-zjf/DeepImpute/ckpt', help='保存模型的路径')
args_parser.add_argument('--seed', type=int, default=1024, help='划分数据使用的随机种子')
args_parser.add_argument('--print_freq', type=int, default=50, help='信息显示的频率')
args_parser.add_argument('--conv_channel_list', type=list, default=[2, 64, 64, 128], help='卷积核的尺寸')
args_parser.add_argument('--linear_size', type=list, default=[512, 256, 256], help='全联接的神经元数量')
args_parser.add_argument('--kernel_size', type=list, default=[3, 5,3,1], help='卷积的核大小')
args_parser.add_argument('--drop_rate', type=float, default=0.5, help='避免过拟合的dropout概率')
args_parser.add_argument('--test_sampling_num', type=int, default=2000, help='测试时数据的大小')
args_parser.add_argument('--datapath', type=str, default='/Users/23mofang_mac/Project_Codes/python/company_project/Impute/data', help='数据路径')
args = args_parser.parse_args()
