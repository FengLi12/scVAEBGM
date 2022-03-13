import os
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import issparse

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

from anndata import AnnData
import scanpy as sc
import episcanpy as epi
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from glob import glob
np.warnings.filterwarnings('ignore')
    
def preprocessing_atac(
        adata, 
        min_genes=100,
        min_cells=0.01, 
        n_top_genes=30000,
        target_sum=None,
        log=None
    ):
    """
    preprocessing
    """
    print('Raw dataset shape: {}'.format(adata.shape))
        
    # adata.X[adata.X>1] = 1    # Doing

    # sc.pp.log1p(adata)

    # Filtering cells
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # Filtering genes
    if min_cells < 1:
        min_cells = min_cells * adata.shape[0]
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Finding variable features
    adata = epi.pp.select_var_feature(adata, nb_features=n_top_genes, show=False, copy=True)

    # Normalizing total per cell
    sc.pp.normalize_total(adata, target_sum=target_sum)

    # TF-IDF transformation
    x = adata.X.toarray().T
    from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
    nfreqs = 1.0 * x / np.tile(np.sum(x, axis=0), (x.shape[0], 1))
    x = nfreqs * np.tile(np.log(1 + 1.0 * x.shape[1] / np.sum(x, axis=1)).reshape(-1, 1), (1, x.shape[1]))
    x = x.T  # (cells, peaks)
    x = MinMaxScaler().fit_transform(x)
    adata.X = x
    adata.X = scipy.sparse.csr_matrix(adata.X)

    print('Processed dataset shape: {}'.format(adata.shape))

    return adata

class SingleCellDataset(Dataset):
    """
    Dataset for dataloader
    """
    def __init__(self, adata):
        self.adata = adata
        self.shape = adata.shape
        
    def __len__(self):
        return self.adata.X.shape[0]
    
    # 重写__getitem__()方法，使其在每个batch中能够返回各行的label(细胞类型)
    def __getitem__(self, idx):
        x = self.adata.X[idx].toarray().squeeze()
        domain_id = self.adata.obs['batch'].cat.codes[idx]
        label = self.adata.obs['label'].cat.codes[idx]
#         return x, domain_id, idx
        return x,label


def load_dataset(
        data_list,
        batch_categories=None,
        batch_key='batch',
        batch_name='batch',
        min_genes=100,
        min_cells=0.01,
        n_top_genes=2000,
        batch_size=64,
        # log=None,
    ):
    """
    Load dataset with preprocessing
    """
    # 读取数据文件
    df = pd.read_csv('data_raw_Forebrain\sc_mat.txt', sep='\t', index_col=0).T
    
    # 读取label文件（新）
    Y = pd.read_csv('data_raw_Forebrain\label.txt', sep='\t', header=None)
    list_label = []
    label = list(Y[0].values)
    for x in label:
        x = x.strip()
        list_label.append(x)
    list_label

    # 将data与label合并，生成带标签数据data
    # data.loc[len(data)] = list_label
    # data = data.rename(index={11285: "label"})
    # data = data.T

    # 生成adata数据
    adata = AnnData(df.values, dict(obs_names=df.index.values), dict(var_names=df.columns.values))

    # 储存为稀疏矩阵
    adata.X = scipy.sparse.csr_matrix(adata.X)
    
    adata.var_names_make_unique()

    if batch_name!='batch':
        adata.obs['batch'] = adata.obs[batch_name]
    if 'batch' not in adata.obs:
        adata.obs['batch'] = 'batch'
    adata.obs['batch'] = adata.obs['batch'].astype('category')

    # 为adata添加label标签
    adata.obs['label'] = list_label
    adata.obs['label'] = adata.obs['label'].astype('category')

    # 数据预处理
    adata = preprocessing_atac(
        adata,
        min_genes=min_genes,
        min_cells=min_cells,
        # log=log,
    )

    scdata = SingleCellDataset(adata) # Wrap AnnData into Pytorch Dataset，（scdata是一行一行的形式）
    
    trainloader = DataLoader(
        scdata, 
        batch_size=batch_size, 
        drop_last=True,
        shuffle=True,
        num_workers=4
    )

    testloader = DataLoader(scdata, batch_size=batch_size, drop_last=False, shuffle=False)
    
    return adata, trainloader, testloader 
