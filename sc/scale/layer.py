import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

import math
import numpy as np


def build_mlp(layers, activation, bn=False, dropout=0):
    """
    Build multilayer linear perceptron
    """
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))

        if bn:
            net.append(nn.BatchNorm1d(layers[i]))
        net.append(activation)
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    # print("\nnet:", net)
    return nn.Sequential(*net)

# 新增
# 参数 activation=nn.Softmax(dim=1)
def build_mlp2(layers, activation=None):
    """
    Build multilayer linear perceptron
    """
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        # net.append(nn.BatchNorm1d(layers[i]))
        net.append(activation)
    return nn.Sequential(*net)

class Encoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0):
        """
        Inference network

        Attempts to infer the probability distribution
        p(z|x) from the data by fitting a variational
        distribution q_φ(z|x). Returns the two parameters
        of the distribution (µ, log σ²).

        :param dims: dimensions of the networks
           given by the number of neurons on the form
           [input_dim, [hidden_dims], latent_dim].
        """
        super(Encoder, self).__init__()

        [x_dim, h_dim, z_dim] = dims
        # self.hidden = build_mlp([x_dim, *h_dim], bn=bn, dropout=dropout)
        self.hidden = build_mlp([x_dim]+h_dim, activation = nn.Tanh(), bn=bn, dropout=dropout)   # [x_dim]+h_dim 是将这两项合并为一个列表，传入build_mlp()的参数其实是: ([11285, 1024, 128], False, 0)
        self.sample = GaussianSample(([x_dim]+h_dim)[-1], z_dim)   # 传入GaussianSample()的参数其实是: (128, 10)
        # self.sample = GaussianSample([x_dim, *h_dim][-1], z_dim)

    def forward(self, x):
        x = self.hidden(x);
        return self.sample(x)


class Decoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0, output_activation=None):
        """
        Generative network

        Generates samples from the original distribution
        p(x) by transforming a latent representation, e.g.
        by finding p_θ(x|z).

        :param dims: dimensions of the networks
            given by the number of neurons on the form
            [latent_dim, [hidden_dims], input_dim].
        """
        super(Decoder, self).__init__()

        [z_dim, h_dim, x_dim] = dims

        # print("\n[z_dim, *h_dim]:", [z_dim, *h_dim])   #[10, 8]
        # print("\nbn:", bn)                             # False
        # print("\ndropout:", dropout)                   # 0
        # self.hidden = build_mlp([z_dim, *h_dim], bn=bn, dropout=dropout)  # [z_dim, *h_dim] 表示将其合并
        self.hidden = build_mlp2([z_dim, *h_dim], activation = nn.Tanh())
#         self.hidden = build_mlp([z_dim]+h_dim, bn=bn, dropout=dropout)

        self.reconstruction = nn.Linear([z_dim, *h_dim][-1], x_dim)
#         self.reconstruction = nn.Linear(([z_dim]+h_dim)[-1], x_dim)
#         print("\nself.reconstruction:", self.reconstruction)  #Linear(in_features=8, out_features=11285, bias=True)

        self.output_activation = output_activation

    def get_hidden(self, x):
        x = self.hidden(x)       # x.shape = (batch size * 8), ji(128 * 8)
        return x

    def forward(self, x):
        x = self.hidden(x)       # x.shape = (batch size * 8), ji(128 * 8)

        if self.output_activation is not None:    # True
            return self.output_activation(self.reconstruction(x))
        else:
            return self.reconstruction(x)

class DeterministicWarmup(object):
    """
    Linear deterministic warm-up as described in
    [Sønderby 2016]. (warmup有助于收敛, 有助于减缓模型在初始阶段对mini-batch的提前过拟合现象，保持分布的平稳
有助于保持模型深层的稳定性)
    """
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1/n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t

    def next(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t


###################
###################
class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """
    def reparametrize(self, mu, logvar):
        epsilon = torch.randn(mu.size(), requires_grad=False, device=mu.device) # torch.randn():用来生成随机数字的tensor，这些随机数字满足标准正态分布（0~1）
        std = logvar.mul(0.5).exp_()
#         std = torch.clamp(logvar.mul(0.5).exp_(), -5, 5)
        z = mu.addcmul(std, epsilon)

        return z

class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)

        return self.reparametrize(mu, log_var), mu, log_var

