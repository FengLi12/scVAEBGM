import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, ReduceLROnPlateau

import time
import math
import numpy as np
import netAE_modularity
from modularity_data import partition_labeled_data
from tqdm import tqdm, trange
from itertools import repeat
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt

from .layer import Encoder, Decoder, build_mlp, DeterministicWarmup
from .loss import elbo, elbo_SCALE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, dims, bn=False, dropout=0, binary=True):  # binary:False是mse平方损失函数，True是二项损失函数，改变还需要改变损失def的返回值类型
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].

        :param dims: x, z and hidden dimensions of the networks
        """
        super(VAE, self).__init__()
        [x_dim, z_dim, encode_dim, decode_dim] = dims    # dims: [11285, 10, [1024, 128], [8]]
        self.binary = binary
        # print("\nbinary:", binary)
        if binary:
            decode_activation = nn.Sigmoid()
            # decode_activation = nn.Softmax(dim=1)
        else:
            decode_activation = None
        # 新增
        # decode_activation = nn.Softmax(dim=1)
        # 原来
        # decode_activation = nn.Sigmoid()

        self.encoder = Encoder([x_dim, encode_dim, z_dim], bn=bn, dropout=dropout)
        # [x_dim, encode_dim, z_dim] = [11285, [3200, 1600, 800, 400], 10]

        self.decoder = Decoder([z_dim, decode_dim, x_dim], bn=bn, dropout=dropout, output_activation=decode_activation)
        # [z_dim, decode_dim, x_dim] = [10, 11285]

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights (利用xavier正态分布来保证，通过网络层时，输入和输出的方差相同，避免梯度消失)
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        :param x: input data
        :return: reconstructed input
        """
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)

        return recon_x

    def loss_modularity1(self, x, labels=[]):
        z, mu, logvar = self.encoder(x)
        # modularity loss:
        labels = list(labels.numpy())
        fea_labels_sets = partition_labeled_data(z, labels)
        q = netAE_modularity.calc_modularity_mult(fea_labels_sets, gamma = 150)
        # netLoss = -q
        # print("\nnetLoss:", netLoss)
        return q

    def loss_classification(self, x, labels=[]):
        z, mu, logvar = self.encoder(x)
        labels = list(labels.numpy())
        labels = torch.tensor(labels)
        d = self.decoder.get_hidden(z)   # d是(64 * 8)
        logsfm = torch.log(d)            # 对softmax()的结果进行一次log
        loss_N = nn.NLLLoss()              # 定义损失函数

        # predi = torch.max(d, 1)
        # print("\n0:", predi[0])  #每行的最大值
        # print("\n1:", predi[1])  #每行的最大值的下标(0-7)
        # loss_fn = nn.CrossEntropyLoss()  # 定义损失函数(不使用，因为其中包含softmax()，与网络结构中的softmax()重复)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        loss = loss_N(logsfm, labels)  # 计算损失

        return loss

    def fit(self, dataloader,
            lr=0.0002,
            weight_decay=5e-4,
            device='cuda:0',
            beta = 1,
            n = 200,
            max_iter=30000,
            outdir='./',
            label=[]
       ):

        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, nesterov=True)
        # optimizer = torch.optim.Adadelta(self.parameters(), lr=lr)
        Beta = DeterministicWarmup(n=n, t_max=beta)

        mol_loss = []  # 模块 loss 列表
        classification_loss = []   # 分类 loss 列表
        classification_acc = []   # 分类 acc 列表
        recon_loss1 = [] # recon损失
        kld_loss = [] # kld损失

        iteration = 0
        n_epoch = int(np.ceil(max_iter/len(dataloader)))  #ceil() 函数返回数字的上入整数
        with tqdm(range(460), total=460, desc='Epochs') as tq:
            for epoch in tq:
#                 epoch_loss = 0
                epoch_recon_loss, epoch_kl_loss, epoch_mol_loss, epoch_cla_loss = 0, 0, 0, 0
                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations')
                for i, x in tk0:
                    # epoch_lr = adjust_learning_rate(lr, optimizer, iteration)
#                     print("\nbatch1:", x[0],"\index2:",x[0][1][11281], "\nshape:",x[0].shape, "\nlabel1:", x[1])
                    x[0] = x[0].float().to(device)
                    optimizer.zero_grad()
                    
                    recon_loss, kl_loss = self.loss_function(x[0])

                    # 先停用模块化和分类损失两项
                    # modul_lose = self.loss_modularity1(x[0],x[1])

                    # cla_loss = self.loss_classification(x[0],x[1])

                    # print("\nmodul_lose:", modul_lose)
#                     loss = (recon_loss + next(Beta) * kl_loss)/len(x[0]);
#                     loss = (recon_loss + kl_loss)/len(x[0])
#                     loss = (recon_loss + kl_loss + modul_lose + cla_loss)/len(x[0])
#                     loss.backward()
#                     print("\nrecon_loss: ", recon_loss, "\nkl_loss: ", kl_loss)
                    loss = (recon_loss + kl_loss)/len(x[0])
                    loss.backward(retain_graph=True)

                    # 停
                    # cla_loss.backward(retain_graph=True)
                    # modul_lose.backward()

                    torch.nn.utils.clip_grad_norm(self.parameters(), 10) # clip (让每一次训练的结果都不过分的依赖某一部分神经元，
                                                                         # 在训练的时候随机忽略一些神经元和神经的链接，使得神经网络变得不完整，
                                                                         # 是解决过拟合的一种方法)
                    optimizer.step()
                    
                    epoch_kl_loss += kl_loss.item()
                    epoch_recon_loss += recon_loss.item()

                    # 停
                    # epoch_mol_loss += modul_lose.item()
                    # epoch_cla_loss += cla_loss.item()

                    tk0.set_postfix_str('loss={:.3f} recon_loss={:.3f} kl_loss={:.3f}'.format(
                            loss, recon_loss/len(x[0]), kl_loss/len(x[0])))
                    tk0.update(1)
                    
                    iteration+=1
                # 停
                # tq.set_postfix_str('recon_loss {:.3f} kl_loss {:.3f} mol_loss {:.4f} classificat_loss={:.4f}'.format(
                #     epoch_recon_loss/((i+1)*len(x[0])), epoch_kl_loss/((i+1)*len(x[0])), epoch_mol_loss*1000000, epoch_cla_loss/(i+1)))
                tq.set_postfix_str('recon_loss {:.3f} kl_loss {:.3f}'.format(
                    epoch_recon_loss/((i+1)*len(x[0])), epoch_kl_loss/((i+1)*len(x[0]))))
                # 停
                # mol_loss.append(epoch_mol_loss)
                # classification_loss.append(epoch_cla_loss)
                recon_loss1.append(epoch_recon_loss)
                kld_loss.append(epoch_kl_loss)
        #（绘制recon损失曲线）
        i = 0
        for i in range(len(recon_loss1)):
            recon_loss1[i] /= 65 * 32
        # print("\nlo:", classification_loss)
        # 绘制 Loss 下降曲线图，loss_list是已存储好loss值的列表，其元素类型是tensor
        from matplotlib import pyplot as plt
        plt.figure(figsize = (8,6))
        l = list(range(0,len(recon_loss1)))
        plt.plot(l, recon_loss1,'r-', label = u'VAE_Net')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("recon")
        plt.show()
        plt.legend()  # 显示图例
        plt.savefig("recon.png")

        #（绘制kld损失曲线）
        i = 0
        for i in range(len(kld_loss)):
            kld_loss[i] /= 65 * 32
        # print("\nlo:", classification_loss)
        # 绘制 Loss 下降曲线图，loss_list是已存储好loss值的列表，其元素类型是tensor
        from matplotlib import pyplot as plt
        plt.figure(figsize = (8,6))
        l = list(range(0,len(kld_loss)))
        plt.plot(l, kld_loss,'b-', label = u'VAE_Net')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("kld")
        plt.show()
        plt.legend()  # 显示图例
        plt.savefig("kld.png")


        # 停（绘制分类损失曲线）
        # i = 0
        # for i in range(len(classification_loss)):
        #     classification_loss[i] /= 65
        # # print("\nlo:", classification_loss)
        # # 绘制 Loss 下降曲线图，loss_list是已存储好loss值的列表，其元素类型是tensor
        # from matplotlib import pyplot as plt
        # plt.figure(figsize = (8,6))
        # l = list(range(0,len(classification_loss)))
        # plt.plot(l, classification_loss,'r-', label = u'RCSCA_Net')
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title("simple")
        # plt.show()
        # plt.legend()  # 显示图例
        # plt.savefig("fig.png")

    def encodeBatch(self, dataloader, device='cpu', out='z', transforms=None):
        output = []
        for x,l in dataloader:
            x = x.view(x.size(0), -1).float().to(device)
            z, mu, logvar = self.encoder(x)

            if out == 'z':
                output.append(z.detach().cpu())
            elif out == 'x':
                recon_x = self.decoder(z)
                output.append(recon_x.detach().cpu().data)
            elif out == 'logit':
                output.append(self.get_gamma(z)[0].cpu().detach().data)

        output = torch.cat(output).numpy()
        return output

class K(VAE):
    def __init__(self, dims, n_centroids):
        super(K, self).__init__(dims)
        self.n_centroids = n_centroids
        # init kk
        self.kk = 0

    # 新加的
    def init_bgmm_params(self, dataloader, device='cpu'):
        """
        Init SCALE model with BGMM model parameters
        """
        bgmm = BayesianGaussianMixture(n_components=self.n_centroids, covariance_type='diag',
                                           max_iter=1000, random_state=0,
                                           weight_concentration_prior_type='dirichlet_distribution',
                                           weight_concentration_prior=53)

        z = self.encodeBatch(dataloader, device)

        bgmm.fit(z)

        ll = bgmm.fit(z)  # 用EM算法估计模型参数。

        self.kk = len(np.unique(ll.predict(z)))

    def get_kk(self):
        return self.kk

class SCALE(K):
    def __init__(self, dims, n_centroids):
        super(SCALE, self).__init__(dims, n_centroids)
        self.n_centroids = n_centroids
        z_dim = dims[1]

        # init c_params
        self.pi = nn.Parameter(torch.ones(n_centroids)/n_centroids)  # pc
        self.mu_c = nn.Parameter(torch.zeros(z_dim, n_centroids)) # mu
        self.var_c = nn.Parameter(torch.ones(z_dim, n_centroids)) # sigma^2

    def loss_function(self, x):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)

        gamma, mu_c, var_c, pi = self.get_gamma(z) #, self.n_centroids, c_params)
        likelihood, kl_loss = elbo_SCALE(recon_x, x, gamma, (mu_c, var_c, pi), (mu, logvar), binary=self.binary)

        return -likelihood, kl_loss

    def get_gamma(self, z):
        """
        Inference c from z

        gamma is q(c|x)
        q(c|x) = p(c|z) = p(c)p(c|z)/p(z)
        """
        n_centroids = self.n_centroids

        N = z.size(0)
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
        pi = self.pi.repeat(N, 1) # NxK
#         pi = torch.clamp(self.pi.repeat(N,1), 1e-10, 1) # NxK
        mu_c = self.mu_c.repeat(N,1,1) # NxDxK
        var_c = self.var_c.repeat(N,1,1) + 1e-8 # NxDxK

        # p(c,z) = p(c)*p(z|c) as p_c_z
        p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5*torch.log(2*math.pi*var_c) + (z-mu_c)**2/(2*var_c), dim=1)) + 1e-10
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma, mu_c, var_c, pi

        # self.mu_c.data.copy_(torch.from_numpy(bgmm.means_.T.astype(np.float32)))
        # self.var_c.data.copy_(torch.from_numpy(bgmm.covariances_.T.astype(np.float32)))

