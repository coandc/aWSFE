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
import sys
sys.path.append('../')
import copy
import argparse
from PIL import Image
try:
    import cPickle as pickle
except:
    import pickle
import math

import modified_resnet
import modified_linear
import modified_resnet_aux
import utils_pytorch
from utils_imagenet.utils_dataset import split_images_labels
from utils_imagenet.utils_dataset import merge_images_labels
from utils_incremental.compute_features import compute_features
from utils_incremental.compute_accuracy import compute_accuracy
from utils_incremental.incremental_train_and_eval import incremental_train_and_eval_two_branches
from utils_incremental.incremental_train_and_eval import balance_finetuning
from utils_incremental.compute_accuracy import compute_accuracy_fusion
from utils_incremental.compute_features import compute_features_fusion

######### Modifiable Settings ##########
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='imagenet_sub', type=str)
parser.add_argument('--datadir', default='data/seed_1993_subset_100_imagenet/data', type=str)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--train_batch_size', default=128, type=int)
parser.add_argument('--test_batch_size', default=100, type=int)
parser.add_argument('--eval_batch_size', default=100, type=int)
parser.add_argument('--num_workers', default=4, type=int, \
    help='the number of workers for loading data')
parser.add_argument('--nb_cl_fg', default=50, type=int, \
    help='the number of classes in first group')
parser.add_argument('--nb_cl', default=10, type=int, \
    help='Classes per group')
parser.add_argument('--nb_protos', default=20, type=int, \
    help='Number of prototypes per class at the end')
parser.add_argument('--nb_runs', default=1, type=int, \
    help='Number of runs (random ordering of classes at each run)')
parser.add_argument('--ckp_prefix', default=os.path.basename(sys.argv[0])[:-3], type=str, \
    help='Checkpoint prefix')
parser.add_argument('--epochs', default=90, type=int, \
    help='Epochs')
parser.add_argument('--T', default=2, type=float, \
    help='Temporature for distialltion')
parser.add_argument('--resume', action='store_true', \
    help='resume from checkpoint')
parser.add_argument('--fix_budget', action='store_true', \
    help='fix budget')
parser.add_argument('--random_seed', default=1993, type=int, \
    help='random seed')
args = parser.parse_args()

########################################
assert(args.nb_cl_fg % args.nb_cl == 0)
assert(args.nb_cl_fg >= args.nb_cl)
train_batch_size       = 128            # Batch size for train
test_batch_size        = 50             # Batch size for test
eval_batch_size        = 128            # Batch size for eval
base_lr                = 0.1            # Initial learning rate
lr_strat               = [30, 60]       # Epochs where learning rate gets decreased
lr_factor              = 0.1            # Learning rate decrease factor
custom_weight_decay    = 1e-4           # Weight Decay
custom_momentum        = 0.9            # Momentum

base_lr2                = 0.1            # Initial learning rate
lr2_strat               = [30, 60]      # Epochs where learning rate gets decreased
lr2_factor              = 0.1            # Learning rate decrease factor

args.ckp_prefix        = '{}_nb_cl_fg_{}_nb_cl_{}_nb_protos_{}'.format(args.ckp_prefix, args.nb_cl_fg, args.nb_cl, args.nb_protos)
np.random.seed(args.random_seed)        # Fix the random seed
print(args)
########################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Data loading code
traindir = os.path.join(args.datadir, 'train')
valdir = os.path.join(args.datadir, 'val')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trainset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
testset =  datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
evalset =  datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))


exemplar_trainset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
balanced_trainset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

# Initialization
dictionary_size     = 1500 # 1500 training images for each class
top1_acc_list_cumul = np.zeros((int(args.num_classes/args.nb_cl),3,args.nb_runs))
top1_acc_list_ori   = np.zeros((int(args.num_classes/args.nb_cl),3,args.nb_runs))


X_train_total, Y_train_total = split_images_labels(trainset.imgs) # shape(X_train_total) = (128856,)
X_valid_total, Y_valid_total = split_images_labels(testset.imgs)

# Launch the different runs
for iteration_total in range(args.nb_runs):
    start_time = time.time()
    # Select the order for the class learning
    order_name = "./checkpoint/seed_{}_{}_order_run_{}.pkl".format(args.random_seed, args.dataset, iteration_total)
    print("Order name:{}".format(order_name))
    if os.path.exists(order_name):
        print("Loading orders")
        order = utils_pytorch.unpickle(order_name)
    else:
        print("Generating orders")
        np.random.seed(args.random_seed)
        order = np.arange(args.num_classes)
        np.random.shuffle(order)
        utils_pytorch.savepickle(order, order_name)
    order_list = list(order)
    print(order_list)

    # Initialization of the variables for this run
    X_valid_cumuls    = []
    X_protoset_cumuls = []
    X_train_cumuls    = []
    Y_valid_cumuls    = []
    Y_protoset_cumuls = []
    Y_train_cumuls    = []
    alpha_dr_herding  = np.zeros((int(args.num_classes/args.nb_cl),dictionary_size,args.nb_cl),np.float32)

    # The following contains all the training samples of the different classes
    # because we want to compare our method with the theoretical case where all the training samples are stored
    # prototypes = np.zeros((args.num_classes,dictionary_size,X_train_total.shape[1],X_train_total.shape[2],X_train_total.shape[3]))
    prototypes = [[] for i in range(args.num_classes)]
    for orde in range(args.num_classes):
        prototypes[orde] = X_train_total[np.where(Y_train_total==order[orde])]
    prototypes = np.array(prototypes)

    start_iter = int(args.nb_cl_fg/args.nb_cl)-1
    acc_list_all_CNN = []
    acc_list_all_iCaRL = []
    acc_list_all_NCM = []
    acc_aver_all_CNN = []
    acc_aver_all_iCaRL = []
    acc_aver_all_NCM = []
    acc_cumul_all = []
    fusion_vars = nn.ParameterList()
    for idx in range(4):
        fusion_vars.append(nn.Parameter(torch.FloatTensor([0.5])))
    fusion_vars.to(device)
    for iteration in range(start_iter, int(args.num_classes/args.nb_cl)):
        start_time1 = time.time()
        #init model
        if iteration == start_iter:
            ############################################################
            last_iter = 0
            ############################################################
            tg_model = modified_resnet.resnet18(num_classes=args.nb_cl_fg)
            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            print("in_features:", in_features, "out_features:", out_features)
            ref_model = None
            aux_model = None
        elif iteration == start_iter+1:
            ############################################################
            last_iter = iteration
            ############################################################
            #increment classes
            ref_model = copy.deepcopy(tg_model)
            ref_fusion_vars = copy.deepcopy(fusion_vars)
            ref_dict = ref_model.state_dict()
            
            aux_model = modified_resnet_aux.resnetaux18(num_classes=args.nb_cl_fg)
            aux_dict = aux_model.state_dict()
            state_dict = {k:v for k,v in ref_dict.items() if k in aux_dict.keys()}
            aux_dict.update(state_dict)
            aux_model.load_state_dict(aux_dict)
            aux_model.to(device)
            
            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            print("in_features:", in_features, "out_features:", out_features)
            new_fc = modified_linear.SplitLinear(in_features, out_features, args.nb_cl)
            new_fc.fc1.weight.data = tg_model.fc.weight.data
            new_fc.fc1.bias.data = tg_model.fc.bias.data
            tg_model.fc = new_fc
            ref_aux_model = None
        else:
            ############################################################
            last_iter = iteration
            ############################################################
            ref_model = copy.deepcopy(tg_model)
            ref_aux_model = copy.deepcopy(aux_model)
            ref_fusion_vars = copy.deepcopy(fusion_vars)
            
            aux_model = modified_resnet_aux.resnetaux18(num_classes=args.nb_cl_fg)
            aux_dict = aux_model.state_dict()
            aux_dict.update(state_dict) # bese model
            aux_model.load_state_dict(aux_dict)
            aux_model.to(device)
            
            in_features = tg_model.fc.in_features
            out_features1 = tg_model.fc.fc1.out_features
            out_features2 = tg_model.fc.fc2.out_features
            print("in_features:", in_features, "out_features1:", \
                out_features1, "out_features2:", out_features2)
            new_fc = modified_linear.SplitLinear(in_features, out_features1+out_features2, args.nb_cl)
            new_fc.fc1.weight.data[:out_features1] = tg_model.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = tg_model.fc.fc2.weight.data
            new_fc.fc1.bias.data[:out_features1] = tg_model.fc.fc1.bias.data
            new_fc.fc1.bias.data[out_features1:] = tg_model.fc.fc2.bias.data
            tg_model.fc = new_fc


        # Prepare the training data for the current batch of classes
        actual_cl        = order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)]
        indices_train_10 = np.array([i in order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)] for i in Y_train_total])
        indices_test_10  = np.array([i in order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)] for i in Y_valid_total])

        X_train          = X_train_total[indices_train_10]
        X_valid          = X_valid_total[indices_test_10]
        X_valid_cumuls.append(X_valid)
        X_train_cumuls.append(X_train)
        X_valid_cumul    = np.concatenate(X_valid_cumuls)
        X_train_cumul    = np.concatenate(X_train_cumuls)

        Y_train          = Y_train_total[indices_train_10]
        Y_valid          = Y_valid_total[indices_test_10]
        Y_valid_cumuls.append(Y_valid)
        Y_train_cumuls.append(Y_train)
        Y_valid_cumul    = np.concatenate(Y_valid_cumuls)
        Y_train_cumul    = np.concatenate(Y_train_cumuls)

        # Add the stored exemplars to the training data
        if iteration == start_iter:
            X_valid_ori = X_valid
            Y_valid_ori = Y_valid
        else:
            X_protoset = np.concatenate(X_protoset_cumuls)
            Y_protoset = np.concatenate(Y_protoset_cumuls)
            # X_train    = np.concatenate((X_train,X_protoset),axis=0)
            # Y_train    = np.concatenate((Y_train,Y_protoset))

        # Launch the training loop
        print('Batch of classes number {0} arrives ...'.format(iteration+1))
        map_Y_train = np.array([order_list.index(i) for i in Y_train])
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
        
        if iteration > start_iter:
            exemplar_map_Y_train = np. array([order_list.index(i) for i in Y_protoset])

        ############################################################
        current_train_imgs = merge_images_labels(X_train, map_Y_train)
        trainset.imgs = trainset.samples = current_train_imgs
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        # old exemplar
        if iteration > start_iter:
            exemplar_current_train_imgs = merge_images_labels(X_protoset, exemplar_map_Y_train)
            exemplar_trainset.imgs = exemplar_trainset.samples = exemplar_current_train_imgs
            exemplar_trainloader = torch.utils.data.DataLoader(exemplar_trainset, batch_size=32, shuffle=True, num_workers=2)
        else:
            exemplar_trainloader = None
        
        current_test_imgs = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
        testset.imgs = testset.samples = current_test_imgs
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
            shuffle=False, num_workers=args.num_workers)
        print('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
        print('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid_cumul), max(map_Y_valid_cumul)))
        if iteration > start_iter:
            print('Max and Min of exemplar train labels: {}, {}'.format(min(exemplar_map_Y_train), max(exemplar_map_Y_train)))
        ##############################################################
        ckp_name = './checkpoint/run{}_{}_iteration_{}_model.pth'.format(iteration_total, args.random_seed,iteration)
        aux_ckp_name = './checkpoint/run{}_{}_iteration_{}_aux_model.pth'.format(iteration_total, args.random_seed,iteration)
        print('ckp_name', ckp_name)
        
        start_time2 = time.time()
        if args.resume and os.path.exists(ckp_name):
            print("###############################")
            print("Loading models from checkpoint")
            tg_model = torch.load(ckp_name)
            if (os.path.exists(aux_ckp_name)) and (aux_model is not None):
                print("Loading aux_models from checkpoint")
                aux_model = torch.load(aux_ckp_name)
            print("###############################")
        else:
            tg_params = tg_model.parameters()
            ###############################
            tg_model = tg_model.to(device)
            if iteration > start_iter:
                ref_model = ref_model.to(device)
                
                aux_params = aux_model.parameters()
                branch2_lr = base_lr2
                branch2_weight_decay = custom_weight_decay 
                aux_optimizer = optim.SGD(aux_params, lr=base_lr2, momentum=custom_momentum, weight_decay=custom_weight_decay)
                aux_lr_scheduler = lr_scheduler.MultiStepLR(aux_optimizer, milestones=lr2_strat, gamma=lr2_factor)
                
                tg_params_new =[{'params': tg_params, 'lr': base_lr, 'weight_decay': custom_weight_decay}, \
                    {'params': aux_params, 'lr': branch2_lr, 'weight_decay': branch2_weight_decay}]
                tg_optimizer = optim.SGD(tg_params_new, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
                tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)
                
                fusion_optimizer = optim.SGD(fusion_vars, lr=1e-8, momentum=custom_momentum, weight_decay=custom_weight_decay)
                fusion_lr_scheduler = lr_scheduler.MultiStepLR(fusion_optimizer, milestones=lr_strat, gamma=lr_factor)
            else:
                tg_optimizer = optim.SGD(tg_params, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
                tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)
            ###############################
            
            # Build the balanced dataloader
            if iteration > start_iter:
                X_train_this_step = X_train_total[indices_train_10]
                Y_train_this_step = Y_train_total[indices_train_10]
                the_idx = np.random.randint(0,len(X_train_this_step),size=args.nb_cl*args.nb_protos)
                X_balanced_this_step = np.concatenate((X_train_this_step[the_idx],X_protoset),axis=0)
                Y_balanced_this_step = np.concatenate((Y_train_this_step[the_idx],Y_protoset),axis=0)
                map_Y_train_this_step = np.array([order_list.index(i) for i in Y_balanced_this_step])
                
                balance_current_train_imgs = merge_images_labels(X_balanced_this_step, map_Y_train_this_step)
                balanced_trainset.imgs = balanced_trainset.samples = balance_current_train_imgs
                balancedloader = torch.utils.data.DataLoader(balanced_trainset, batch_size=args.test_batch_size, \
                shuffle=False, num_workers=args.num_workers)
                
            
            tg_model = incremental_train_and_eval_two_branches(args.epochs, fusion_vars, ref_fusion_vars, tg_model, aux_model, ref_model, ref_aux_model, tg_optimizer, tg_lr_scheduler, fusion_optimizer, fusion_lr_scheduler, trainloader, exemplar_trainloader, testloader, balancedloader, iteration, start_iter, args) 
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(tg_model, ckp_name)
            # torch.save(aux_model, aux_ckp_name)

        # balanced finetuning
        if iteration > start_iter:
            X_train_this_step = X_train_total[indices_train_10]
            Y_train_this_step = Y_train_total[indices_train_10]
            the_idx = np.random.randint(0,len(X_train_this_step),size=args.nb_cl*args.nb_protos)
            X_balanced_this_step = np.concatenate((X_train_this_step[the_idx],X_protoset),axis=0)
            Y_balanced_this_step = np.concatenate((Y_train_this_step[the_idx],Y_protoset),axis=0)
            map_Y_train_this_step = np.array([order_list.index(i) for i in Y_balanced_this_step])
            balance_current_train_imgs = merge_images_labels(X_balanced_this_step, map_Y_train_this_step)
            balanced_trainset.imgs = balanced_trainset.samples = balance_current_train_imgs
            balancedloader = torch.utils.data.DataLoader(balanced_trainset, batch_size=args.test_batch_size, \
            shuffle=False, num_workers=args.num_workers)
            
            fc_params = list(map(id, tg_model.fc.parameters()))
            ignored_params = filter(lambda p: id(p) not in fc_params, tg_model.parameters())
            fc_params_new =[{'params': tg_model.fc.parameters(), 'lr': 0.00001, 'weight_decay': custom_weight_decay}, 
                            {'params': ignored_params, 'lr': 0, 'weight_decay': 0}]
            fc_optimizer = optim.SGD(fc_params_new, lr=0.0001, momentum=custom_momentum, weight_decay=custom_weight_decay)
            fc_lr_scheduler = lr_scheduler.MultiStepLR(fc_optimizer, milestones=[15, 30], gamma=lr_factor)
            tg_model = balance_finetuning(args.epochs, fusion_vars, tg_model, aux_model, ref_model, fc_optimizer, fc_lr_scheduler, testloader, balancedloader, iteration, start_iter, args) 
        
        ### Exemplars
        if args.fix_budget:
            nb_protos_cl = int(np.ceil(args.nb_protos*args.num_classes*1.0/args.nb_cl/(iteration+1)))
        else:
            nb_protos_cl = args.nb_protos
        tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
        num_features = tg_model.fc.in_features
        # Herding
        start_time3 = time.time()
        print('Updating exemplar set...')
        for iter_dico in range(last_iter*args.nb_cl, (iteration+1)*args.nb_cl):
            current_eval_set = merge_images_labels(prototypes[iter_dico], np.zeros(len(prototypes[iter_dico])))
            evalset.imgs = evalset.samples = current_eval_set
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                shuffle=False, num_workers=args.num_workers, pin_memory=True)
            num_samples = len(prototypes[iter_dico])            
            mapped_prototypes = compute_features_fusion(args, fusion_vars, tg_model, aux_model, start_iter, iteration, tg_feature_model, evalloader, num_samples, num_features)
            D = mapped_prototypes.T
            D = D/np.linalg.norm(D,axis=0)

            # Herding procedure : ranking of the potential exemplars
            mu  = np.mean(D,axis=1)
            index1 = int(iter_dico/args.nb_cl)
            index2 = iter_dico % args.nb_cl
            alpha_dr_herding[index1,:,index2] = alpha_dr_herding[index1,:,index2]*0
            w_t = mu
            iter_herding     = 0
            iter_herding_eff = 0
            while not(np.sum(alpha_dr_herding[index1,:,index2]!=0)==min(nb_protos_cl,500)) and iter_herding_eff<1000:
                tmp_t   = np.dot(w_t,D)
                ind_max = np.argmax(tmp_t)
                iter_herding_eff += 1
                if alpha_dr_herding[index1,ind_max,index2] == 0:
                    alpha_dr_herding[index1,ind_max,index2] = 1+iter_herding
                    iter_herding += 1
                w_t = w_t+mu-D[:,ind_max]

        # Prepare the protoset
        X_protoset_cumuls = []
        Y_protoset_cumuls = []

        # Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
        print('Computing mean-of_exemplars and theoretical mean...')
        class_means = np.zeros((num_features, args.num_classes, 2))
        for iteration2 in range(iteration+1):
            for iter_dico in range(args.nb_cl):
                current_cl = order[range(iteration2*args.nb_cl,(iteration2+1)*args.nb_cl)]
                current_eval_set = merge_images_labels(prototypes[iteration2*args.nb_cl+iter_dico], \
                    np.zeros(len(prototypes[iteration2*args.nb_cl+iter_dico])))
                evalset.imgs = evalset.samples = current_eval_set
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                    shuffle=False, num_workers=args.num_workers, pin_memory=True)
                num_samples = len(prototypes[iteration2*args.nb_cl+iter_dico])
                mapped_prototypes = compute_features_fusion(args, fusion_vars, tg_model, aux_model, start_iter, iteration, tg_feature_model, evalloader, num_samples, num_features)
                D = mapped_prototypes.T
                D = D/np.linalg.norm(D,axis=0)
                # Flipped version also
                # evalset.test_data = prototypes[iteration2*args.nb_cl+iter_dico][:,:,:,::-1].astype('uint8')
                # evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                #     shuffle=False, num_workers=2)
                # mapped_prototypes2 = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                # D2 = mapped_prototypes2.T
                # D2 = D2/np.linalg.norm(D2,axis=0)
                D2 = D

                # iCaRL
                alph = alpha_dr_herding[iteration2,:,iter_dico]
                assert((alph[num_samples:]==0).all())
                alph = alph[:num_samples]
                alph = (alph>0)*(alph<nb_protos_cl+1)*1.
                X_protoset_cumuls.append(prototypes[iteration2*args.nb_cl+iter_dico][np.where(alph==1)[0]])
                Y_protoset_cumuls.append(order[iteration2*args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))
                alph = alph/np.sum(alph)
                class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
                class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])

                # Normal NCM
                # alph = np.ones(dictionary_size)/dictionary_size
                alph = np.ones(num_samples)/num_samples
                class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
                class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])

        torch.save(class_means, \
            './checkpoint/{}_run_{}_iteration_{}_class_means.pth'.format(args.ckp_prefix,iteration_total, iteration))

        start_time4 = time.time()
        current_means = class_means[:, order[range(0,(iteration+1)*args.nb_cl)]]
        ##############################################################
        # Calculate validation error of model on the first nb_cl classes:
        map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
        print('Computing accuracy on the original batch of classes...')
        current_eval_set = merge_images_labels(X_valid_ori, map_Y_valid_ori)
        evalset.imgs = evalset.samples = current_eval_set
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        ori_acc = compute_accuracy_fusion(args, aux_model, iteration, start_iter, fusion_vars, tg_model, tg_feature_model, current_means, evalloader)
        top1_acc_list_ori[iteration, :, iteration_total] = np.array(ori_acc).T
        
        acc_list_each_CNN = [0] * (int((args.num_classes - args.nb_cl_fg)/args.nb_cl + 1))
        acc_list_each_iCaRL = [0] * (int((args.num_classes - args.nb_cl_fg)/args.nb_cl + 1))
        acc_list_each_NCM = [0] * (int((args.num_classes - args.nb_cl_fg)/args.nb_cl + 1))
        if iteration == start_iter:
            acc_list_each_CNN[0] = ori_acc[0]
            acc_list_each_iCaRL[0] = ori_acc[1]
            acc_list_each_NCM[0] = ori_acc[2]
        if iteration > start_iter:
            for i in range(iteration-start_iter+1):
                # Calculate validation error of model on the each classes set:
                if i == 0:
                    print('Computing accuracy on the previous 0 ~ {} classes...'.format(args.nb_cl_fg))
                else:
                    print('Computing accuracy on the previous {} ~ {} classes...'.format((i-1)*args.nb_cl + args.nb_cl_fg, i*args.nb_cl + args.nb_cl_fg))
                map_Y_valid_prev = np.array([order_list.index(i) for i in Y_valid_cumuls[i]])
                current_eval_set = merge_images_labels(X_valid_cumuls[i], map_Y_valid_prev)
                evalset.imgs = evalset.samples = current_eval_set
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size,
                        shuffle=False, num_workers=args.num_workers)
                # each_acc = compute_accuracy(tg_model, tg_feature_model, current_means, evalloader)
                each_acc = compute_accuracy_fusion(args, aux_model, iteration, start_iter, fusion_vars, tg_model, tg_feature_model, current_means, evalloader)
                acc_list_each_CNN[i] = each_acc[0]
                acc_list_each_iCaRL[i] = each_acc[1]
                acc_list_each_NCM[i] = each_acc[2]
        acc_list_all_CNN.append(acc_list_each_CNN)
        acc_list_all_iCaRL.append(acc_list_each_iCaRL)
        acc_list_all_NCM.append(acc_list_each_NCM)
        
        ##############################################################
        # Calculate validation error of model on the cumul of classes:
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
        print('Computing cumulative accuracy...')
        current_eval_set = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
        evalset.imgs = evalset.samples = current_eval_set
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size,
                shuffle=False, num_workers=args.num_workers)        
        cumul_acc = compute_accuracy_fusion(args, aux_model, iteration, start_iter, fusion_vars, tg_model, tg_feature_model, current_means, evalloader)
        acc_cumul_all.append(cumul_acc[0])
        acc_aver_all_iCaRL.append(cumul_acc[1])
        acc_aver_all_NCM.append(cumul_acc[2])
        top1_acc_list_cumul[iteration, :, iteration_total] = np.array(cumul_acc).T
        
        ##############################################################
        print("————————————————— Pretrain finished in {}s.".format(start_time2 - start_time1))
        print("————————————————— Training process finished in {}s.".format(start_time3 - start_time2))
        print("————————————————— Herding and computing the class means finished in {}s.".format(start_time4 - start_time3))
        print("————————————————— Computing accuracy finished in {}s.".format(int(time.time() - start_time4)))   
        print("Task {} finished in {}s.".format(iteration+1, int(time.time() - start_time1)))
    print('The average incremental accuracy is {} ({})'.format(np.mean(acc_cumul_all), acc_cumul_all,))
    print('Accuracy of each task in each incremental step is:')
    print('CNN: {}'.format(acc_list_all_CNN))
    print('\nThe average incremental accuracy of iCaRL is {} ({})'.format(np.mean(acc_aver_all_iCaRL), acc_aver_all_iCaRL))
    print('iCaRL: {}'.format(acc_list_all_iCaRL))
    print('\nThe average incremental accuracy of NCM is {} ({})'.format(np.mean(acc_aver_all_NCM), acc_aver_all_NCM))
    print('NCM: {}'.format(acc_list_all_NCM))

    # Final save of the data
    torch.save(top1_acc_list_ori, \
        './checkpoint/{}_run_{}_top1_acc_list_ori.pth'.format(args.ckp_prefix, iteration_total))
    torch.save(top1_acc_list_cumul, \
        './checkpoint/{}_run_{}_top1_acc_list_cumul.pth'.format(args.ckp_prefix, iteration_total))
    print("Training finished in {}s.".format(int(time.time() - start_time)))