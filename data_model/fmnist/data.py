from torchvision.datasets import FashionMNIST
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
dirname = os.path.dirname(__file__)

import sys
sys.path.append(os.path.join(dirname, '../'))
from data_utils import _get_partitioner, _use_partitioner, CustomImageFolder

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

def get_dataset(ranks:list, workers:list, batch_size:int, full_batch_size:int=None, data_aug:bool=True, dataset_root='./dataset', **kwargs):
    if data_aug:
        trainset = FashionMNIST(root=dataset_root + '/fashion_mnist_data', train=True, download=True, transform=transform)
        testset = FashionMNIST(root=dataset_root + '/fashion_mnist_data', train=False, download=True, transform=transform)
    else:
        trainset = FashionMNIST(root=dataset_root + '/fashion_mnist_data', train=True, download=True)
        testset = FashionMNIST(root=dataset_root + '/fashion_mnist_data', train=False, download=True)
    
    partitioner = _get_partitioner(trainset, workers, **kwargs)
    data_ratio_pairs = {}
    data_ratio_pairs_full_batch = {}
    for rank in ranks:
        data, ratio = _use_partitioner(partitioner, rank, workers)
        # full_batch_size = len(data)
        # print(1, len(data.indices), data.indices)
        full_batch_size = len(data) if full_batch_size is None or full_batch_size <= 0 else full_batch_size
        print(rank, "full batch size", full_batch_size)

        data_mini_batch = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)
        data_ratio_pairs[rank] = (data_mini_batch, ratio)
        
        data_full_batch = DataLoader(dataset=data, batch_size=full_batch_size, shuffle=False)
        data_ratio_pairs_full_batch[rank] = (data_full_batch, ratio)
        
    testset = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

    return data_ratio_pairs, testset, data_ratio_pairs_full_batch

def get_dataset_with_precat(ranks:list, workers:list, batch_size:int, test_required:bool=False, dataset_root='./dataset'):
    if test_required:
        try:
            testset = get_testset_from_folder(batch_size, dataset_root)
        except:
            testset = get_testdataset(batch_size, dataset_root=dataset_root)
        finally:
            testset = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    else:
        testset = None

    data_ratio_pairs = {}
    for rank in ranks:
        idx = np.where(workers == rank)[0][0]
        current_path = dataset_root + '/fashion_mnist_data/{}_partitions/{}'.format(len(workers), idx)
        trainset = CustomImageFolder(root=current_path, transform=transform)
        trainset = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False)
        with open(current_path + '/weight.txt', 'r') as f:
            ratio = eval(f.read())
        data_ratio_pairs[rank] = (trainset, ratio)
    
    return data_ratio_pairs, testset

def get_testdataset(batch_size:int, dataset_root='./dataset'):
    testset = FashionMNIST(root=dataset_root + '/fashion_mnist_data', train=False, download=True, transform=transform)
    testset = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    return testset

def get_testset_from_folder(batch_size:int, dataset_root='./dataset'):
    current_path = dataset_root + '/fashion_mnist_data/testset'
    testset = CustomImageFolder(root=current_path, transform=transform)
    testset = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    return testset

if __name__ == "__main__":
    # store partitioned dataset 
    num_workers, bsz = 10, 1
    workers = np.arange(num_workers) + 1
    path = 'D:/dataset'
    
    # data_ratio_pairs, testset = get_dataset(workers, workers, bsz, isNonIID=False, dataset_root=path, data_aug=False)
    data_ratio_pairs, testset = get_dataset(workers, workers, bsz, isNonIID=True, isDirichlet=True, alpha=0.1, dataset_root=path, data_aug=False)
    path = path + '/fashion_mnist_data/{}_partitions'.format(num_workers)
    if os.path.exists(path) is False:
        os.makedirs(path)

    data_ratio_pairs["testset"] = (testset, 0.0)
    for idx, pair in data_ratio_pairs.items():
        data, ratio = pair
        data = data.dataset
        current_path = os.path.join(path, str(idx)) if idx != "testset" else os.path.join(path, "../testset")
        if os.path.exists(current_path):
            import shutil
            shutil.rmtree(current_path)
        os.makedirs(current_path)

        with open(current_path + '/weight.txt', 'w') as f:
            f.write('{}\t{}\n'.format(idx, ratio))
        
        for i in range(len(data)):
            sample, target = data[i]
            if os.path.exists(os.path.join(current_path, str(int(target)))) is False:
                os.makedirs(os.path.join(current_path, str(int(target))))
            sample.save(current_path + '/{}/{}.jpg'.format(target, i))
