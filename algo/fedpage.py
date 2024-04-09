from copy import deepcopy
import sys
import torch.nn as nn
from torch.optim import SGD

def train_iter(rank, criterion, w_optimizer, last_model, w_model, w_model_backup, 
    data_ratio_pairs, data_ratio_pairs_full_batch, last_global_gradients, args, device):

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
    epoch_batch_cnt = 0

    correct = 0
    total = 0
    
    # last_model = deepcopy(w_model)
    last_optimizer = SGD(last_model.parameters(), lr=args.lr)
    last_model.train()

    learning_rate = args.lr
    print("lr", learning_rate)
    param_grad = deepcopy(last_global_gradients)
    
    K = 0

    maximum_steps = args.K * len(data_ratio_pairs[rank][0]) if args.epoch else args.K

    print("step", maximum_steps)

    completed_steps = -1
    train_data = iter(data_ratio_pairs[rank][0])
    train_data_full = iter(data_ratio_pairs_full_batch[rank][0])
    while completed_steps < maximum_steps:
        if completed_steps == maximum_steps:
            break
        
        # K = batch_idx
        try:
            if completed_steps == -1:
                data, target = next(train_data_full)
            else:
                data, target = next(train_data)
        except:
            train_data = iter(data_ratio_pairs[rank][0])
            data, target = next(train_data)
        
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
        # print(rank, batch_id, 'Acc: %.3f%% (%d/%d)'
        #       % (100. * correct / total, correct, total), loss.data.item(), time.time() - batch_start_time)
        sys.stdout.flush()

    delta_ws = []

    for p_idx, param in enumerate(w_model_backup.parameters()):
        delta_ws.append((param - list(w_model.parameters())[p_idx].data) / (maximum_steps * learning_rate))

    return delta_ws, (epoch_train_loss / epoch_batch_cnt), correct / total