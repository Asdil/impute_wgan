# -*- coding : utf-8 -*-

import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append('..')
from impute_gan import Generator, Discriminator
from dataset import RandomDataset

def test_generator_module(args):
    generator = Generator(args)
    print(generator)
    x = torch.FloatTensor(args.batch_size, args.dim).uniform_(-1,1)
    m = torch.FloatTensor(args.batch_size, args.dim).uniform_(-1,1)
    z = torch.FloatTensor(args.batch_size, args.dim).uniform_(-1,1)
    rate = np.random.uniform(-1, 1, (args.dim))
    out = generator(x,m,z, rate)
    assert(out.size() == (args.batch_size, args.dim))

def test_torch_mul():
    x = torch.FloatTensor(args.batch_size, args.dim).uniform_(-1,1)
    w1 = nn.Parameter(torch.FloatTensor(args.dim).uniform_(0, 1))
    print(x.size(), w1.size())
    print(((x+x)*w1).size())

def test_discriminator_module(args):
    discriminator = Discriminator(args)
    x = torch.FloatTensor(args.batch_size, args.dim).uniform_(-1,1)
    m = torch.FloatTensor(args.batch_size, args.dim).uniform_(-1,1)
    z = torch.FloatTensor(args.batch_size, args.dim).uniform_(-1,1)
    h = torch.FloatTensor(args.batch_size, args.dim).uniform_(-1,1)
    out = discriminator(x, m, z, h)
    assert(out.size() == (args.batch_size, args.dim))

def test_dataset():
    dataset = RandomDataset(args)
    sampleM = dataset.sample_M(args.batch_size, args.dim, args.miss_p)
    trainset, testset, start, rare_indexes, rate = dataset.load_snp()
    print(trainset.shape, testset.shape, start, rare_index.shape, rare)





if __name__ == '__main__':
    from args import args
    test_discriminator_module(args)
    #test_torch_mul()