import sys
import torch
import torch.nn as nn
from copy import deepcopy
from torch.optim import SGD
import algo.fedamd as fedamd_algo
import math

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

def train_iter(rank, criterion, w_optimizer, w_model, w_model_backup, 
    data_ratio_pairs, global_direction, args, device):

    w_model.train()

    # --- disable nomalization layers ---
    for module in w_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    epoch_train_loss = 0
    epoch_batch_cnt = 0

    correct = 0
    total = 0
    
    last_model = deepcopy(w_model)
    last_optimizer = SGD(last_model.parameters(), lr=args.lr)
    last_model.train()
    
    learning_rate = args.lr
    param_grad = deepcopy(global_direction)

    total_batch = len(data_ratio_pairs[rank][0])

    maximum_steps = args.K * len(data_ratio_pairs[rank][0]) if args.epoch else args.K

    print("step", maximum_steps)

    completed_steps = 0
    print("lr", learning_rate, "bsz cnt", total_batch)

    while completed_steps < maximum_steps:
    
        for batch_idx, (data, target) in enumerate(data_ratio_pairs[rank][0]):
            if completed_steps == maximum_steps:
                break
            
            data, target = data.to(device), target.to(device)
            last_optimizer.zero_grad()
            last_output = last_model(data)
            last_loss = criterion(last_output, target)
            last_loss.backward()

            w_optimizer.zero_grad()
            output = w_model(data)
            loss = criterion(output, target)
            loss.backward()

            # param_grad=[]  #一个client的所有梯度
            for p_idx, param in enumerate(w_model.parameters()):
                
                param_grad[p_idx] = param_grad[p_idx] - list(last_model.parameters())[p_idx].grad.data.clone().detach() \
                    + list(w_model.parameters())[p_idx].grad.data.clone().detach()
                
            last_model = deepcopy(w_model)
            last_optimizer = SGD(last_model.parameters(), lr=args.lr)
            
            for p_idx, param in enumerate(last_model.parameters()):
                param.data = list(w_model.parameters())[p_idx].data.clone().detach()
            
            for p_idx, param in enumerate(w_model.parameters()):
                param.data -= learning_rate * param_grad[p_idx]

            # accuracy calculation
            epoch_train_loss += loss.data.item()
            epoch_batch_cnt += 1
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            completed_steps += 1
            sys.stdout.flush()

    delta_ws = []

    for p_idx, param in enumerate(w_model_backup.parameters()):
        delta_ws.append(param - list(w_model.parameters())[p_idx].data)

    return delta_ws, (epoch_train_loss / epoch_batch_cnt), correct / total

def compute_grad(criterion, w_model, data_target, device):
    w_model.train()

    # --- disable nomalization layers ---
    for module in w_model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    epoch_train_loss = 0
    
    data, target = data_target

    data, target = data.to(device), target.to(device)
    w_model.zero_grad()
    output = w_model(data)
    loss = criterion(output, target)
    loss.backward(retain_graph = True)

    # accuracy calculation
    epoch_train_loss += loss.data.item()
    
    delta_ws = []

    for p_idx, param in enumerate(w_model.parameters()):
        grad_tmp = param.grad.data.clone().detach()
        delta_ws.append(grad_tmp)

    return delta_ws, epoch_train_loss

def run(workers, model, args, data_ratio_pairs, data_ratio_pairs_full_batch, test_data, cpu, gpu):
    worker_num = len(workers)

    model = model.cuda(gpu)
    iterators = [iter(train_data) for train_data, _ in data_ratio_pairs.values()]    
    print('Worker {} successfully received the model. '.format(list(workers)))

    # --- initialization ---

    full_batch_grad = []
    pr = 0
    
    hist_acc = []
    
    # --- training ---
    S, iteration = args.T, 0
    fullbsz = len(data_ratio_pairs_full_batch[0][0].dataset) if args.fullbsz <= 0 else args.fullbsz

    if args.K == 1:
        maximum_steps = 640 / args.bsz
    elif args.K == 2:
        maximum_steps = 320 / args.bsz
    elif args.K == 3:
        maximum_steps = 1280 / args.bsz
    else:
        maximum_steps = args.K

    for s in range(S):
        if iteration > args.T:
            break
        
        # get global full batch

        # --- random sampling ---
        import random
        part_list = [i for i in range(worker_num)]
        random.shuffle(part_list)
        part_list = part_list[:args.num_part]
        print(s * 2, part_list)
  
        full_batch_grad = []
        
        for worker in part_list:
            mymodel = deepcopy(model)
            criterion = torch.nn.CrossEntropyLoss()

            train_data = iter(data_ratio_pairs_full_batch[worker][0])
            grad, _ = compute_grad(criterion, mymodel, next(train_data), gpu)
            full_batch_grad.append(grad)

        print("finish full batch", s) 

        # compute the initial orientation
        global_direction = []
        for p_idx, param in enumerate(model.parameters()):
            param_dir = torch.zeros_like(param.data)
            for worker in part_list:
                param_dir += (full_batch_grad[part_list.index(worker)][p_idx] / len(part_list))
            
            global_direction.append(param_dir.clone().detach())
        full_batch_grad = [deepcopy(global_direction) for _ in part_list]
        
        last_model = deepcopy(model)
        for t in range(math.ceil(1 + fullbsz/(args.K * args.bsz))):
            model.train()

            # update local stored orientation 
            trainset_order = {}
            for worker in part_list:
                trainset_order[worker] = []
                completed_steps = 0
                while completed_steps < maximum_steps:
                    for batch_idx, (data, target) in enumerate(data_ratio_pairs[worker][0]):
                        if completed_steps == maximum_steps:
                            break
                        mymodel, mylastmodel = deepcopy(model), deepcopy(last_model)
                        trainset_order[worker].append((deepcopy(data), deepcopy(target)))

                        criterion = torch.nn.CrossEntropyLoss()
                        model_grad, _ = compute_grad(criterion, mymodel, (data, target), gpu)
                        lastmodel_grad, _ = compute_grad(criterion, mylastmodel, (data, target), gpu)
                        
                        for idx, (a, b) in enumerate(zip(model_grad, lastmodel_grad)):
                            full_batch_grad[part_list.index(worker)][idx] += ((a-b)/(maximum_steps*args.bsz))

                        completed_steps += 1

                trainset_order[worker] = (trainset_order[worker], 0)
                
            global_direction = []
            for p_idx, param in enumerate(model.parameters()):
                param_dir = torch.zeros_like(param.data)
                for worker in part_list:
                    param_dir += (full_batch_grad[part_list.index(worker)][p_idx] / len(part_list))
                
                global_direction.append(param_dir.clone().detach())

            tot_loss = 0
            tot_acc = 0
            grads = []
            
            # --- local iteration ---
            worker = random.choice(part_list)
            
            mymodel = deepcopy(model)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = SGD(mymodel.parameters(), lr=args.lr)

            grad, loss, acc = train_iter(worker, criterion, optimizer, mymodel, model, 
                    data_ratio_pairs, global_direction, args, gpu)
            tot_loss += loss
            tot_acc += acc
            grads.append(grad)
            
            print('Worker: {}\tCommunition Rounds: {}\tLoss: {}'.format(worker, s * 2 + t, loss))
            
            # --- aggregation & model updates ---
            if len(grads) > 0:        
                tot_loss /= len(grads)
                tot_acc /= len(grads)
                iteration = iteration + 1

                if math.isnan(tot_loss):
                    print("NaN occurs. Break....")
                    return 
                
                print("Round: {}\tTotal loss: {}".format(iteration, tot_loss))

                for p_idx, param in enumerate(model.parameters()):
                    grads_avg = torch.zeros_like(param.data)
                    for i in range(len(grads)):    
                        grads_avg += (grads[i][p_idx] / len(grads)) 

                    param.data -= grads_avg
                    # param.data -= grads[pick_client][p_idx]
            
            test_loss, test_acc = check_accuracy(iteration, test_data, model, gpu)
            hist_acc.append(test_acc)
            print(iteration, "Acc mean", sum(hist_acc)/len(hist_acc))
            # if sum(hist_acc)/len(hist_acc) <= 0.17 and t > 150:
            #     print('Low accuracy. Not good. ')
            #     return 
