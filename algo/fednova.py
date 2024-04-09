import sys 
import torch.nn as nn

def train_iter(rank, criterion, w_optimizer, w_model, w_model_backup, 
    data_ratio_pairs, args, device):

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

    maximum_steps = args.K * len(data_ratio_pairs[rank][0]) if args.epoch else args.K

    print("step", maximum_steps)
    
    completed_steps = 0
    while completed_steps < maximum_steps:
    
        for batch_idx, (data, target) in enumerate(data_ratio_pairs[rank][0]):
            
            if completed_steps == maximum_steps:
                break

            data, target = data.to(device), target.to(device)
            w_optimizer.zero_grad()
            output = w_model(data)
            # print(output, target)
            loss = criterion(output, target)
            loss.backward()
            w_optimizer.step()
            
            # accuracy calculation
            epoch_train_loss += loss.data.item()
            epoch_batch_cnt += 1

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            # print(predicted, target)
            completed_steps += 1
            # print(rank, batch_idx, 'Acc: %.3f%% (%d/%d)'
                #   % (100. * correct / total, correct, total), loss.data.item())
            sys.stdout.flush()

    delta_ws = []

    for p_idx, param in enumerate(w_model_backup.parameters()):
        delta_ws.append((param - list(w_model.parameters())[p_idx].data) / completed_steps)

    return delta_ws, (epoch_train_loss / epoch_batch_cnt), correct / total
