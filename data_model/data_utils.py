# from random import Random
from copy import deepcopy
import random
import numpy as np
from torchvision.datasets.folder import ImageFolder

""" Return the sample with its original label (must be an integer) """
class CustomImageFolder(ImageFolder):
    def __init__(self, **kwargs):
        super(CustomImageFolder, self).__init__(**kwargs)
        self.custom_target_transform, self.target_transform = self.target_transform, None

    def __getitem__(self, index):
        sample, target = self.__getitem__(index)
        target = int(self.classes[target])
        if self.custom_target_transform is not None:
            target = self.custom_target_transform(target)
        return sample, target

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.indices = deepcopy(index)
        # print(0, len(self.indices), self.indices)
        self.targets = []
        for idx in self.indices:
            _, target = self.data[idx]
            self.targets.append(target)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        return self.data[data_idx]

'''
    Data partitioner
    :param data: imgs from dataset such as CIFAR10 
    :param sizes: 
    :param isNonIID: True indicates the data are heterogeneous 
    :param isDirichlet: True indicates the non-iid data distribution follows Dirichlet 
    :param alpha: positive real number for dirichlet distribution while positive integer for pathological 
'''
class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], isNonIID=True, isDirichlet=False, alpha=3, seed=1234):
        self.data = data
        if isNonIID:
            if isDirichlet:
                self.partitions, self.ratio = self.__getDirichletData__(data, sizes, seed, alpha)
            else:
                self.partitions, self.ratio = self.__getPathologicalData__(data, sizes, seed, alpha)

        else:
            self.partitions = {}
            self.ratio = sizes
            # random.seed(seed)
            data_len = len(data) 
            indexes = [x for x in range(0, data_len)] 
            random.shuffle(indexes) 
            
            for idx, frac in enumerate(sizes): 
                part_len = int(frac * data_len)
                self.partitions[idx] = indexes[0:part_len]
                indexes = indexes[part_len:]

    def use(self, partition):
        # print(partition, self.partitions[partition])
        return Partition(self.data, self.partitions[partition]), self.ratio[partition]

    def __getDirichletData__(self, data, psizes, seed, alpha):
        n_nets = len(psizes)
        labelList = np.array(data.targets)
        K = len(np.unique(labelList))
        min_size = 0
        N = len(labelList)
        # np.random.seed(seed)

        net_dataidx_map = {}
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(labelList == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
        print('Data statistics: %s' % str(net_cls_counts))

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes/np.sum(local_sizes)
        # print(weights)

        return net_dataidx_map, weights

    def __getPathologicalData__(self, data, size, seed, classes):
        labelList = np.array(data.targets)
        K = len(np.unique(labelList))
        # np.random.seed(seed)

        # print(size)
        classes = -1 if len(size) * classes < K and len(size) < K else max(min(classes, K), 1)
        all_labels = np.arange(K)
        if classes == -1:
            worker_labels = np.array_split(all_labels, len(size))
        else: 
            worker_labels = np.array([], dtype=int)
            for _ in range(int(len(size) * classes / K)):
                worker_labels = np.concatenate([worker_labels, np.random.choice(all_labels, size=K, replace=False)])
            worker_labels = np.array_split(worker_labels, len(size))
            print('worker_labels', worker_labels)
            # worker_labels = np.concatenate([np.random.choice(all_labels, size=K, replace=False), 
            #                                 np.random.choice(all_labels, size=len(size)*classes-K, 
            #                                                  replace=True)]).reshape(len(size), classes)
            # # in case that some sets exist duplicate 
            # for idx, _ in enumerate(worker_labels):
            #     while len(np.unique(worker_labels[idx])) != classes:
            #         worker_labels[idx] = np.random.choice(all_labels, classes, replace=False)
        
        data_dict = {}
        for k in range(K):
            data_dict[k] = np.where(labelList == k)[0]
            np.random.shuffle(data_dict[k])
        
        net_dataidx_map = {}

        for idx, _ in enumerate(size):
            net_dataidx_map[idx] = []
            for label in worker_labels[idx]:
                num_label = len(np.where(np.concatenate(worker_labels) == label)[0])
                a = len(np.where(labelList == label)[0]) // num_label
                net_dataidx_map[idx] += list(data_dict[label][0:a])
                data_dict[label] = data_dict[label][a:]
            np.random.shuffle(net_dataidx_map[idx])
        
        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
        # print('Data statistics: %s' % str(net_cls_counts))

        local_sizes = []
        for i, _ in enumerate(size):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes/np.sum(local_sizes)
        # print(weights)

        return net_dataidx_map, weights

def _get_partitioner(dataset, workers:list, isNonIID=True, isDirichlet=False, alpha=3):
    """ Partitioning Data """
    workers_num = len(workers)
    partition_sizes = [1.0 / workers_num for _ in range(workers_num)]

    partitioner = DataPartitioner(dataset, partition_sizes, isNonIID, isDirichlet, alpha)

    return partitioner

def _use_partitioner(partitioner, rank, workers:list):
    return partitioner.use(np.where(workers == rank)[0][0])

def _get_dataset(rank, dataset, workers:list, isNonIID=True, isDirichlet=False, alpha=3):
    partitioner = _get_partitioner(dataset, workers, isNonIID, isDirichlet, alpha)
    return _use_partitioner(partitioner, rank, workers)