#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from utils_pytorch import *
from utils_incremental.process_fusion import process_fusion_feature

def incremental_train_and_eval_two_branches(epochs, fusion_vars, ref_fusion_vars, tg_model, b2_model, ref_model, ref_b2_model, tg_optimizer, tg_lr_scheduler, fusion_optimizer, fusion_lr_scheduler, trainloader, exemplar_trainloader, testloader, balancedloader, iteration, start_iteration, args, fix_bn=False, weight_per_class=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if iteration > start_iteration:
        ref_model.eval()
    for epoch in range(epochs):
        #train
        tg_model.train()
        b2_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    #m.weight.requires_grad = False
                    #m.bias.requires_grad = False
        train_loss = 0
        correct = 0
        total = 0
        tg_lr_scheduler.step()
        fusion_lr_scheduler.step()
        print('\nEpoch: %d, LR: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr())
        start_time = time.time()
        if iteration == start_iteration:
            iterator = trainloader
            for batch_idx, (inputs, targets) in enumerate(iterator):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.long()
                tg_optimizer.zero_grad()
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                loss.backward()
                tg_optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        else:
            iterator = zip(trainloader, exemplar_trainloader)
        
            for batch_idx, (curr, prev) in enumerate(iterator):
                num_old_classes = args.nb_cl * iteration
                
                data, target = curr
                target = target - num_old_classes # note here
                batch_size = data.shape[0]
                
                data_r, target_r = prev
                replay_size = data_r.shape[0]
                data, data_r = data.to(device), data_r.to(device)
                
                input = torch.cat((data,data_r))
                target, target_r = target.to(device), target_r.to(device)
                target, target_r = target.long(), target_r.long()
                
                outputs, _ = process_fusion_feature(args, fusion_vars, tg_model, b2_model, input) 
                
                loss_KD = 0
                loss_CE_curr = 0
                loss_CE_prev = 0

                # loss_CE as SS-IL
                curr = outputs[:batch_size, num_old_classes:]
                loss_CE_curr = nn.CrossEntropyLoss(reduction='sum')(curr, target)
                prev = outputs[batch_size:batch_size+replay_size, :num_old_classes]
                loss_CE_prev = nn.CrossEntropyLoss(reduction='sum')(prev, target_r)
                loss_CE = (loss_CE_curr + loss_CE_prev) / (batch_size + replay_size)

                
                # loss_KD as SS-IL
                if iteration == start_iteration+1:
                    score = ref_model(input).data
                else:
                    ref_b2_model.eval()
                    score, _ = process_fusion_feature(args, ref_fusion_vars, ref_model, ref_b2_model, input)
                    score = score.data
                loss_KD = torch.zeros(iteration - start_iteration).cuda()
                for t in range(iteration - start_iteration):
                    if t == 0:
                        start_KD = 0
                        end_KD = args.nb_cl_fg
                    else:
                        start_KD = args.nb_cl_fg + args.nb_cl * (t-1)
                        end_KD = args.nb_cl_fg + args.nb_cl * t

                    soft_target = F.softmax(score[:,start_KD:end_KD] / args.T, dim=1)
                    output_log = F.log_softmax(outputs[:,start_KD:end_KD] / args.T, dim=1)
                    loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean')  * (args.T**2)
                    # if t == 0:
                    #     loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean')  * args.nb_cl_fg / (iteration * args.nb_cl)  * (args.T**2)
                    # else:
                    #     loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean')  * args.nb_cl / (iteration * args.nb_cl)  * (args.T**2)
                loss_KD = loss_KD.sum()
                
                loss = loss_CE + loss_KD
                tg_optimizer.zero_grad()
                loss.backward()
                tg_optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                target = target + num_old_classes
                targets = torch.cat((target, target_r))
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        if iteration == start_iteration:
            print('Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format(\
                len(iterator), train_loss/(batch_idx+1), 100.*correct/total))
        else:
            print('Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format(len(trainloader)+len(exemplar_trainloader), train_loss/(batch_idx+1), 100.*correct/total))
        
        # adapted aggregation weights
        # tg_model.eval()
        # b2_model.eval()        
        # for batch_idx, (inputs, targets) in enumerate(balancedloader):
        #     targets = targets.long()
        #     if batch_idx <= 500:
        #         inputs, targets = inputs.to(device), targets.to(device)
        #         outputs, _ = process_fusion_feature(args, fusion_vars, tg_model, b2_model, inputs)
        #         loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
        #         loss.backward()
        #         fusion_optimizer.step()
        # if args.dataset == 'cifar100':
        #     print('The fusion weight for tg_moedl is: [{:.4f}, {:.4f}, {:.4f}]'.format(fusion_vars[0].item(), fusion_vars[1].item(), fusion_vars[2].item()))
        # elif args.dataset == 'imagenet_sub' or args.dataset == 'imagenet':
        #     print('The fusion weight for tg_moedl is: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(fusion_vars[0].item(), fusion_vars[1].item(), fusion_vars[2].item(), fusion_vars[3].item()))
        start_time1 = time.time() 

        #eval
        tg_model.eval()
        b2_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.long()
                # outputs = tg_model(inputs)
                outputs, _ = process_fusion_feature(args, fusion_vars, tg_model, b2_model, inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format(\
            len(testloader), test_loss/(batch_idx+1), 100.*correct/total))
        print("Training: {}s, All finished: {}s".format(int(start_time1 - start_time), int(time.time() - start_time)))

    return tg_model



def balance_finetuning(epochs, fusion_vars, tg_model, b2_model, ref_model, fc_optimizer, fc_lr_scheduler, testloader, balancedloader, iteration, start_iteration, args, fix_bn=False, weight_per_class=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if iteration > start_iteration:
        ref_model.eval()
    
    print('\nBalance finetuning begin:')
    for epoch in range(30):
        start_time = time.time()
        tg_model.eval()
        b2_model.eval()
        fc_lr_scheduler.step()
        print('\nEpoch: %d, LR: ' % epoch, end='')
        print(fc_lr_scheduler.get_lr())        
        for batch_idx, (inputs, targets) in enumerate(balancedloader):
            targets = targets.long()
            if batch_idx <= 500:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = process_fusion_feature(args, fusion_vars, tg_model, b2_model, inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                loss.backward()
                fc_optimizer.step()
        
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.long()
                # outputs = tg_model(inputs)
                outputs, _ = process_fusion_feature(args, fusion_vars, tg_model, b2_model, inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format(\
            len(testloader), test_loss/(batch_idx+1), 100.*correct/total))
        print("All of this epoch finished in {}s.".format(int(time.time() - start_time)))

    return tg_model