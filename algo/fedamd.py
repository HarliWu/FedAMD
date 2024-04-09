from copy import deepcopy
import sys
import torch
import torch.nn as nn
from torch.optim import SGD


def fedamd_train_iter(rank, criterion, w_optimizer:torch.optim.SGD, w_model, w_model_backup, 
    data_ratio_pairs, global_direction, args, device):

    w_model.train()

    # --- disable nomalization layers ---
    for module in w_model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(True)
                # module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(True)
                # module.bias.requires_grad_(False)
            module.eval()

    epoch_train_loss = 0
    epoch_batch_cnt = 0

    correct = 0
    total = 0
    
    last_model = deepcopy(w_model)
    last_optimizer = SGD(last_model.parameters(), lr=args.lr)
    last_model.train()
    
    learning_rate = w_optimizer.defaults['lr']
    param_grad = deepcopy(global_direction)

    total_batch = len(data_ratio_pairs[rank][0])

    maximum_steps = args.K * len(data_ratio_pairs[rank][0]) if args.epoch else args.K

    print("lr", learning_rate, "bsz cnt", total_batch, "step", maximum_steps)
    
    completed_steps = 0
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
                try:
                    param_grad[p_idx] = param_grad[p_idx] - list(last_model.parameters())[p_idx].grad.data.clone().detach() \
                                                        + list(w_model.parameters())[p_idx].grad.data.clone().detach()
                    # param_grad[p_idx] = list(w_model.parameters())[p_idx].grad.data.clone().detach()
                except:
                    param_grad[p_idx] = torch.zeros_like(param_grad[p_idx])
            
            # last_optimizer.zero_grad()
            # w_optimizer.zero_grad()

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
            # print(rank, batch_id, 'Acc: %.3f%% (%d/%d)'
            #       % (100. * correct / total, correct, total), loss.data.item(), time.time() - batch_start_time)
            sys.stdout.flush()

    delta_ws = []

    for p_idx, param in enumerate(w_model_backup.parameters()):
        if args.lr_global_divided:
            delta_ws.append((param - list(w_model.parameters())[p_idx].data) / completed_steps)
        else:
            delta_ws.append(param - list(w_model.parameters())[p_idx].data)

    return delta_ws, (epoch_train_loss / epoch_batch_cnt), correct / total

def compute_grad(rank, criterion, w_model: nn.Module, data_ratio_pairs_full_batch, device):
    w_model.train()

    # --- disable nomalization layers ---
    for module in w_model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(True)
                # module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(True)
                # module.bias.requires_grad_(False)
            module.eval()

    epoch_train_loss = 0
    epoch_batch_cnt = 0

    correct = 0
    total = 0
    train_data = iter(data_ratio_pairs_full_batch[rank][0])
    data, target = next(train_data)

    data, target = data.to(device), target.to(device)
    w_model.zero_grad()
    output = w_model(data)
    loss = criterion(output, target)
    loss.backward(retain_graph = True)

    # accuracy calculation
    epoch_train_loss += loss.data.item()
    
    delta_ws = []

    for p_idx, param in enumerate(w_model.parameters()):
        try:
            grad_tmp = param.grad.data.clone().detach()
            delta_ws.append(grad_tmp)
        except:
            delta_ws.append(torch.zeros_like(param.data))

    return delta_ws, epoch_train_loss

def comp_avg_grad(rank, criterion, w_model: nn.Module, lr, data_ratio_pairs, device):
    w_model.train()
    
    delta_ws = [torch.zeros_like(param) for param in w_model.parameters()]
    tot_data = 0
    epoch_loss = 0.0
    optimizer = torch.optim.SGD(w_model.parameters(), lr=lr)
    
    for (data, target) in data_ratio_pairs[rank][0]:
        data, target = data.to(device), target.to(device)
        tot_data = tot_data + 1
        
        optimizer.zero_grad()
        output = w_model(data)
        loss = criterion(output, target)
        epoch_loss = epoch_loss + loss
        loss.backward()
        
        for p_idx, param in enumerate(w_model.parameters()):
            delta_ws[p_idx] = delta_ws[p_idx] + param.grad.data.clone().detach()
            
        optimizer.step()
        
    delta_ws = [param/tot_data for param in delta_ws]
    epoch_loss = epoch_loss/tot_data
    
    return delta_ws, epoch_loss