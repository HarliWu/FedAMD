from cProfile import label
import numpy as np
from copy import deepcopy
import argparse
import math

import torch
from torch.optim import SGD

import sys
import random

def fedamd_arguments():
    parser = argparse.ArgumentParser(description='Model Training')
    # Method description
    parser.add_argument('--method', type=str, default='FedAvg', help='Running algorithm')
    
    # Dataset & Model 
    parser.add_argument('--root', type=str, default='./dataset', help='The root of dataset')
    parser.add_argument('--dataset', type=str, default='fmnist', help='The name of dataset used')
    parser.add_argument('--model', type=str, default='LeNet', help='The name of model used') 
    parser.add_argument('--presplit', action='store_true', help='Use the split dataset as training')
    parser.add_argument('--non-iid', action='store_true', help='The distribution of training data')
    parser.add_argument('--dirichlet', action='store_true', help='Non-iid distribution follows Dirichlet')
    parser.add_argument('--dir-alpha', type=float, default=0.1, help='The alpha value for dirichlet distrition')
    parser.add_argument('--pathological', action='store_true', help='Non-iid distribution follows Pathological')
    parser.add_argument('--classes', type=int, default=2, help='Number of classes on each client')

    # Training
    parser.add_argument('--bsz', type=int, default=64, help='Batch size for training dataset')
    parser.add_argument('--fullbsz', type=int, default=0, help='Full batch size for training dataset')
    parser.add_argument('--num-part', type=int, default=10, help='Number of partipants')
    parser.add_argument('--num-clients', type=int, default=20, help='Number of clients')
    parser.add_argument('--seed', type=int, default=2022, help='Seed for randomization')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate at local')
    parser.add_argument('--lr-global', type=float, default=1.0, help='learning rate at server')
    parser.add_argument('--T', type=int, default=200, help='Communication rounds')
    parser.add_argument('--K', type=int, default=10, help='Local Communication rounds')
    parser.add_argument('--pr', type=float, default=0.2, help='Probability')
    parser.add_argument('--ts', action='store_true', help='Time series, pr=1: 1/t; pr=2: 1/log t')
    # parser.add_argument('--K', type=int, default=100, help='Local iterations')

    # GPU setting
    parser.add_argument('--gpu-idx', type=int, default=4, help='GPU index')

    args = parser.parse_args()
    return args


def check_accuracy(epoch, loader, model, device):
    
    model.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(epoch, batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    
    model.train()

    return test_loss / (batch_idx + 1), correct/total



def run(workers, model, args, data_ratio_pairs, data_ratio_pairs_full_batch, test_data, cpu, gpu):
    worker_num = len(workers)

    model = model.cuda(gpu)
    last_model = None
    iterators = [iter(train_data) for train_data, _ in data_ratio_pairs.values()]    
    print('Worker {} successfully received the model. '.format(list(workers)))

    # --- initialization ---

    import algo.fedavg as fedavg
    import algo.fedamd as fedamd_algo
    import algo.fedpage as fedpage
    import algo.scaffold as scaffold

    full_batch_grad = []
    pr = args.pr
    if args.method == "FedAMD" or args.method == "SCAFFOLD":    
        for worker in workers:
            mymodel = deepcopy(model)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = SGD(mymodel.parameters(), lr=args.lr)

            grad, _ = fedamd_algo.compute_grad(worker, criterion, optimizer, mymodel, model, 
                data_ratio_pairs_full_batch, gpu)
            full_batch_grad.append(grad)
            print("finish full batch", worker)
        
        if args.method == "FedAMD" and pr >= 1.0:
            pr = int(pr)
    elif args.method == "FedPAGE":
        pr = 0.5
    
    last_global_gradients = []
    for p_idx, param in enumerate(model.parameters()):
        last_global_gradients.append(torch.zeros_like(param.data))
    
    hist_acc = []

    # --- training ---
    for t in range(args.T):
        
        model.train()
        
        # --- random sampling ---
        import random
        part_list = [i for i in range(worker_num)]
        random.shuffle(part_list)
        part_list = part_list[:args.num_part]
        print(t, part_list)
        
        tot_loss = 0.0
        tot_acc = 0.0
        grads = []

        # --- local iteration ---
        
        if args.method == "FedAMD":            
            # --- time series [deprecated] ---
            if args.ts:
                pr = min(1/math.log(max((t+14)/5, math.e)), 1.0)
            print(pr)

            # --- fedamd algorithm ---
            global_direction = []
            for p_idx, param in enumerate(model.parameters()):
                param_dir = torch.zeros_like(param.data)
                for worker in workers:
                    param_dir += (full_batch_grad[worker][p_idx] / worker_num)
                global_direction.append(param_dir.clone().detach())
            

            for worker in part_list:
                
                mymodel = deepcopy(model)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = SGD(mymodel.parameters(), lr=args.lr)

                if (pr < 1 and np.random.choice([True, False], p=[pr, 1-pr])) or (pr > 1 and t % pr == 0):
                    # Anchor group 
                    grad, _ = fedamd_algo.compute_grad(worker, criterion, optimizer, mymodel, model, 
                        data_ratio_pairs_full_batch, gpu)
                    full_batch_grad[worker] = grad
                    print("update full batch direction", worker)
                else:
                    # Miner group 
                    grad, loss, acc = fedamd_algo.fedamd_train_iter(worker, criterion, optimizer, mymodel, model, 
                        data_ratio_pairs, global_direction, args, gpu)
                    
                    tot_loss += loss
                    tot_acc += acc
                    grads.append(grad)
                    print('Worker: {}\tCommunition Rounds: {}\tLoss: {}'.format(worker, t, loss))

        elif args.method == "FedPAGE":
            # --- FedPage ---
            grads = []
            if t == 0 or np.random.choice([True, False], p=[pr, 1-pr]):
                # full updates
                print("full update round", t)
                for worker in workers:
                    mymodel = deepcopy(model)
                    criterion = torch.nn.CrossEntropyLoss()
                    optimizer = SGD(mymodel.parameters(), lr=args.lr)
                    grad, loss = fedamd_algo.compute_grad(worker, criterion, optimizer, mymodel, model, 
                        data_ratio_pairs_full_batch, gpu)
                    tot_loss += loss
                    
                    grads.append(grad)
                    print('Worker: {}\tCommunition Rounds: {}\tFull Loss: {}'.format(worker, t, loss))
                    
                last_model = deepcopy(model)
            else:
                for worker in part_list:
                    mymodel = deepcopy(model)
                    criterion = torch.nn.CrossEntropyLoss()
                    optimizer = SGD(mymodel.parameters(), lr=args.lr)
                    
                    grad, loss, acc = fedpage.train_iter(worker, criterion, optimizer, last_model, mymodel, model, 
                        data_ratio_pairs, data_ratio_pairs_full_batch, last_global_gradients, args, gpu)      

                    tot_loss += loss
                    tot_acc += acc
                    grads.append(grad)
                    print('Worker: {}\tCommunition Rounds: {}\tLoss: {}'.format(worker, t, loss))
                    
                last_model = deepcopy(model)

        elif args.method == "FedAvg":
            # --- FedAvg ---
            for worker in part_list:
                mymodel = deepcopy(model)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = SGD(mymodel.parameters(), lr=args.lr)

                grad, loss, acc = fedavg.train_iter(worker, criterion, optimizer, mymodel, model, 
                        data_ratio_pairs, args, gpu)
                tot_loss += loss
                tot_acc += acc
                grads.append(grad)

                print('Worker: {}\tCommunition Rounds: {}\tLoss: {}'.format(worker, t, loss))

        elif args.method == "SCAFFOLD":
            # --- SCAFFOLD ---

            global_direction = []
            for p_idx, param in enumerate(model.parameters()):
                param_dir = torch.zeros_like(param.data)
                for worker in workers:
                    param_dir += (full_batch_grad[worker][p_idx] / worker_num)
                global_direction.append(param_dir.clone().detach())
            
            for worker in part_list:
                mymodel = deepcopy(model)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = SGD(mymodel.parameters(), lr=args.lr)

                grad, next_dire, loss, acc = scaffold.train_iter(worker, criterion, optimizer, mymodel, model, 
                        data_ratio_pairs, global_direction, full_batch_grad[worker], args, gpu)
                tot_loss += loss
                tot_acc += acc
                grads.append(grad)
                full_batch_grad[worker] = deepcopy(next_dire)

                print('Worker: {}\tCommunition Rounds: {}\tLoss: {}'.format(worker, t, loss))

        else:
            print('No the algorithm mentioned')
            return
                   
        # --- aggregation & model updates ---
        if len(grads) > 0:   
            tot_loss /= len(grads)
            tot_acc /= len(grads)
            
            if math.isnan(tot_loss):
                print("NaN occurs. Break....")
                return 

            print("Round: {}\tTotal loss: {}".format(t, tot_loss))
            
            for p_idx, param in enumerate(model.parameters()):
                grads_avg = torch.zeros_like(param.data)
                for i in range(len(grads)):    
                    grads_avg += (grads[i][p_idx] / len(grads)) 
                
                param.data -= args.lr_global * grads_avg
                # update last gradients
                if args.method == "FedPAGE":
                    last_global_gradients[p_idx] = grads_avg.clone().detach() * args.lr_global
                else:
                    last_global_gradients[p_idx] = grads_avg.clone().detach()

            test_loss, test_acc = check_accuracy(t, test_data, model, gpu)
            hist_acc.append(test_acc)
            
            print(t, "Acc mean", np.mean(hist_acc))
            if np.mean(hist_acc) <= 0.17 and t > 150:
                print('Low accuracy. Not good. ')
                return 

if __name__ == "__main__":
    args = fedamd_arguments()
    print(args)

    data_tuple = []

    # set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    import importlib
    import sys
    sys.path.insert(1, '../')
    dataset = importlib.import_module('data_model.{}.data'.format(args.dataset))
    model = importlib.import_module('data_model.{}.model'.format(args.dataset))
    model = getattr(model, args.model)()

    test_data = dataset.get_testdataset(args.bsz, dataset_root=args.root)
    
    workers = np.arange(args.num_clients)
    worker_num = args.num_clients
    
    alpha = args.dir_alpha if args.dirichlet else args.classes

    train_data_list = []
    print("clients", args.num_clients)
   
    ranks = np.array_split(workers, 1)[0]
    data_ratio_pairs, _, data_ratio_pairs_full_batch = dataset.get_dataset(ranks=ranks, workers=workers, batch_size=args.bsz, 
                                            full_batch_size=args.fullbsz, dataset_root=args.root, isNonIID=args.non_iid, 
                                            isDirichlet=args.dirichlet, alpha=alpha)
    train_data_list = data_ratio_pairs
    train_data_list_full_batch = data_ratio_pairs_full_batch
    
    np.random.seed(args.seed)
    
    cpu = torch.device('cpu')
    gpu = torch.device('cuda:{}'.format(args.gpu_idx))
    print(gpu)
    
    if args.method == "BVRLSGD":
        import algo.bvrlsgd as bvrlsgd
        bvrlsgd.run(workers, model, args, train_data_list, train_data_list_full_batch, test_data, cpu, gpu)
    else:
        run(workers, model, args, train_data_list, train_data_list_full_batch, test_data, cpu, gpu)

    