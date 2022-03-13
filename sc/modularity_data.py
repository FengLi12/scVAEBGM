# 数据集划分方法
import torch
def gen_idx_byclass(labels):
    """
    Neatly organize indices of labeled samples by their classes.

    Parameters
    ----------
    labels : list
        Note that labels should be a simple Python list instead of a tensor.

    Returns
    -------
    idx_byclass : dictionary {[class_label (int) : indices (list)]}
    """
    # print("in gen_idx_byclass...")
    from collections import Counter
    classes = Counter(labels).keys()  # obtain a list of classes
    idx_byclass = {}

    for class_label in classes:
        # Find samples of this class:
        class_idx = []  # indices for samples that belong to this class
        for idx in range(len(labels)):
            if labels[idx] == class_label:
                class_idx.append(idx)
        idx_byclass[class_label] = class_idx

    return idx_byclass


def partition_labeled_data(data, labels):
    """ Partition the labeled dataset with more than 2 labels into groups of binary sets.
    e.g. Dataset with labels (0, 1, 2, 3) is partitioned into (0, (1, 2, 3) as 1), (1 as 0, (2, 3) as 1), (2 as 0, 3 as 1)
    This procedure allows for easy calculation of modularity for binary classes. See the Newman paper for details.

    Parameters
    ----------
    data : tensor (nsamples, nfeatures)
    labels : list

    Returns
    -------
    partitions : list of lists [partition1, partition2, partition3, ...]
        partition1 = [data of partition1, labels of partition1]
    """
    partitions = []  # initialize

    # # 调用gen_idx_byclass()函数得到 (字符串类型-index) 的字典
    # idx_byclass = gen_idx_byclass(labels)
    # # 把字典key值记录到li列表中
    # li = list(idx_byclass.keys())
    # # 改变key值，把字符串类型转换成对应的数字类型。
    # for k in range(len(idx_byclass.keys())):
    #     idx_byclass[k] = idx_byclass.pop(li[k])
    idx_byclass = gen_idx_byclass(labels)

    list_tem = []
    for k, v in idx_byclass.items():
        i = 0
        tem = torch.empty(0).to('cuda')
        for i in range(len(v)):
            d = data[v[i]]
            d = d.unsqueeze(0)
            tem = torch.cat((tem, d), 0)

        label = [k] * len(v)
        label = torch.tensor(label)
        label = label.unsqueeze(1)

        list_tem.append([tem, label])
    list_tem
    return list_tem