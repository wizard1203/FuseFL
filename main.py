#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import copy
from copy import deepcopy
import os
import math
import shutil
import sys
import warnings
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import pdb
import logging
import time

import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))


from helpers.datasets import partition_data, get_image_size, get_num_of_labels
from helpers.utils import get_dataset, average_weights, DatasetSplit, KLDiv, setup_seed, test, progressive_test
from helpers.exp_path import ExpTool


from models.generator import Generator
from models.nets import (CNNCifar, CNNMnist, CNNCifar100, 
                        make_CNNCifar_seqs, make_CNNCifar_Head_seqs)
from models.pnn import PNN
from models.pnn_cnn import PNN_CNN, pnn_resnet18, pnn_resnet50

from models.fl_pnn import Federated_PNN
from models.fl_pnn_cnn import Federated_PNN_CNN, fl_pnn_resnet18, fl_pnn_resnet50
from models.mlp import MLP, make_MLP_seqs, make_MLP_Head_seqs, mlp2, mlp3
from models.fl_exnn import (MLP_Block, CNN_Block,
    merge_layer, Federated_EXNN, Federated_EXNNLayer_global, Federated_EXNNLayer_local,
    fl_exnn_resnet18, fl_exnn_resnet50, 
)
from models.seq_model import Sequential_SplitNN, ReconMIEstimator, LinearProbes
from models.configs import Split_Configs, EXNN_Split_Configs

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from models.resnet import (resnet18, resnet50, 
            resnet18_layers, resnet50_layers, 
            resnet18_head, resnet50_head, make_ResNetMIEstimator, get_res18_out_channels)
from models.vit import deit_tiny_patch16_224
import wandb

from models.auxiliary_nets import Decoder, AuxClassifier

warnings.filterwarnings('ignore')
upsample = torch.nn.Upsample(mode='nearest', scale_factor=7)

from locals.fedavg import LocalUpdate
from locals.fl_progressive import FedPnnLocalUpdate
from locals.progressive import PnnLocalUpdate
from locals.fl_expandable import FedEXNNLocalUpdate
from locals.ccvr import (compute_classes_mean_cov, generate_virtual_representation,
    calibrate_classifier, get_means_covs_from_client)

from alg_train import Ensemble, pretrain, progressive, fed_progressive, fed_expandable, init_fedexnn_merged
from utils import seq_map_values, batch, accuracy, show_model_layers

from helpers.meter import AverageMeter

from tsne_draw import draw_tsne




def str2bool(v):
    if isinstance(v, bool):
        return v
    # if v.lower() in ('yes', 'true', 't', 'y', '1'):
    if isinstance(v, str) and v.lower() in ('true', 'True'):
        return True
    elif isinstance(v, str) and v.lower() in ('false', 'False'):
        return False
    else:
        return v
        # raise argparse.ArgumentTypeError('Boolean value expected.')


def logging_config(args, process_id):
    # customize the log format
    while logging.getLogger().handlers:
        logging.getLogger().handlers.clear()
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    logger.info(args)
    return logger



def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=100,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--gpu', type=int, required=False, default=0)
    parser.add_argument('--num_classes', type=int, default=10, help='.')
    parser.add_argument('--sample_per_class', type=int, default=5000, help='.')
    parser.add_argument('--num_layers', type=int, default=2, help='.')
    parser.add_argument('--mlp_hidden_features', type=int, default=100, help='.')
    parser.add_argument('--cnn_hidden_features', type=int, default=128, help='.')
    parser.add_argument('--res_base_width', type=int, default=64, help='.')
    parser.add_argument('--res_group_norm', type=int, default=0, help='.')


    # Data Free
    parser.add_argument('--adv', default=0, type=float, help='scaling factor for adv loss')

    parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
    parser.add_argument('--save_dir', default='run/synthesis', type=str)
    parser.add_argument('--partition', default='dirichlet', type=str)
    parser.add_argument('--alpha', default=0.5, type=float,
                        help=' If alpha is set to a smaller value, '
                            'then the partition is more unbalanced')

    # Basic
    parser.add_argument('--lr_g', default=1e-3, type=float,
                        help='initial learning rate for generation')
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--g_steps', default=20, type=int, metavar='N',
                        help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=256, type=int, metavar='N',
                        help='number of total iterations in each epoch')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)

    # Misc
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--type', default="pretrain", type=str,
                        help='.')
    parser.add_argument('--main_task', default="train", type=str,
                        help='.')   # train, MI, 
    parser.add_argument('--model', default="", type=str,
                        help='.')
    parser.add_argument('--other', default="", type=str,
                        help='.')
    parser.add_argument('--logging_level', default="INFO", type=str,
                        help='.')
    parser.add_argument('--debug', default="False", type=str,
                        help='.')
    parser.add_argument('--debug_show_exnn_id', default="False", type=str,
                        help='.')
    # 'INFO' or 'DEBUG'

    # federated progressive
    parser.add_argument('--progressive_classifer', default="fixed", type=str,
                        help='.') # fixed, progressive

    # federated expandable NN
    parser.add_argument('--fedexnn_classifer', default="avg", type=str,
                        help='.') #   fixed   multihead
    parser.add_argument('--fedexnn_adapter', default="avg", type=str,
                        help='.') 
    parser.add_argument('--fedexnn_split_num', default=2, type=int,
                        help='.') 
    parser.add_argument('--fedexnn_hetero_layer_depth', default="False", type=str,
                        help='.') 
    parser.add_argument('--fedexnn_self_dropout', default=0.0, type=float,
                        help='.') 
    parser.add_argument('--fedexnn_adapter_constrain_beta', default=0.0, type=float,
                        help='.') 

    # split related 
    parser.add_argument('--split_train', default="False", type=str,
                        help='.') 
    parser.add_argument('--split_local_module_num', default=2, type=int,
                        help='.') 
    parser.add_argument('--split_measure_local_module_num', default=2, type=int,
                        help='.') 
    parser.add_argument('--infopro', default=2, type=int,
                        help='.') 
    parser.add_argument('--MI_cos_lr', default="False", type=str,
                        help='.') 

    # contrastive train
    parser.add_argument('--contrastive_train', default="False", type=str,
                        help='.')
    parser.add_argument('--contrastive_n_views', default=2, type=int,
                        help='.')
    parser.add_argument('--contrastive_weight', default=1.0, type=float,
                        help='.')
    parser.add_argument('--contrastive_projection_dim', default=64, type=int,
                        help='.')

    # backdoor train
    parser.add_argument('--backdoor_train', default="False", type=str,
                        help='.')
    parser.add_argument('--backdoor_n_clients', default=1, type=int,
                        help='.')
    parser.add_argument('--backdoor_size', default=10, type=int,
                        help='.')


    parser.add_argument('--checkpoint', default='no', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')


    # spurious related 
    parser.add_argument('--spufeat', default="", type=str,
                        help='.') 
    parser.add_argument('--aux_net_config', default='1c2f', type=str,
                        help='architecture of auxiliary classifier / contrastive head '
                            '(default: 1c2f; 0c1f refers to greedy SL)'
                            '[0c1f|0c2f|1c1f|1c2f|1c3f|2c2f]')
    parser.add_argument('--local_loss_mode', default='contrast', type=str,
                        help='ways to estimate the task-relevant info I(x, y)'
                            '[contrast|cross_entropy]')
    parser.add_argument('--aux_net_widen', default=1.0, type=float,
                        help='widen factor of the two auxiliary nets (default: 1.0)')
    parser.add_argument('--aux_net_feature_dim', default=0, type=int,
                        help='number of hidden features in auxiliary classifier / contrastive head '
                            '(default: 128)')
    parser.add_argument('--ixx_1', default=0.0, type=float,)   # \lambda_1 for 1st local module
    parser.add_argument('--ixy_1', default=0.0, type=float,)   # \lambda_2 for 1st local module

    parser.add_argument('--ixx_2', default=0.0, type=float,)   # \lambda_1 for (K-1)th local module
    parser.add_argument('--ixy_2', default=0.0, type=float,)   # \lambda_2 for (K-1)th local module

    # EstMI
    parser.add_argument('--EstMI_method', default="infopro", type=str,
                        help='number of local modules (1 refers to end-to-end training)')
    parser.add_argument('--EstFeatNorm', default="no", type=str, help='')
    parser.add_argument('--SaveFeats', default="no", type=str, help='')
    parser.add_argument('--TSNE', default="no", type=str, help='')
    parser.add_argument('--TSNE_points', default=500, type=int, help='')


    # wandb, exp record related
    parser.add_argument("--wandb_offline", type=str, default="True")
    parser.add_argument("--wandb_console", type=str, default="False")
    parser.add_argument("--wandb_entity", type=str, default="your-wandb-entity")
    parser.add_argument("--wandb_key", type=str, default=None)

    parser.add_argument("--exp_abs_path", type=str, default=".")
    parser.add_argument("--project_name", type=str, default="your-wandb-project")
    parser.add_argument("--exp_name", type=str, default="OneShot-FL")
    parser.add_argument("--override_cmd_args", action="store_true")
    parser.add_argument("--tag", type=str, default="debug")
    parser.add_argument("--exp_tool_init_sub_dir", type=str, default="no")

    parser.add_argument("--enable_wandb", type=str, default="False")


    args = parser.parse_args()
    for key in args.__dict__.keys():
        args.__dict__[key] = str2bool(args.__dict__[key])
    return args



def kd_train(synthesizer, model, criterion, optimizer):
    student, teacher = model
    student.train()
    teacher.eval()
    description = "loss={:.4f} acc={:.2f}%"
    total_loss = 0.0
    correct = 0.0
    with tqdm(synthesizer.get_data()) as epochs:
        for idx, (images) in enumerate(epochs):
            optimizer.zero_grad()
            images = images
            with torch.no_grad():
                t_out = teacher(images)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())

            loss_s.backward()
            optimizer.step()

            total_loss += loss_s.detach().item()
            avg_loss = total_loss / (idx + 1)
            pred = s_out.argmax(dim=1)
            target = t_out.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / len(synthesizer.data_loader.dataset) * 100

            epochs.set_description(description.format(avg_loss, acc))


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)


def get_data_info(args):
    if args.dataset == "mnist":
        image_size = 28
        linear_in_feautres = image_size * image_size * 1
        channels = 1
    elif args.dataset == "fmnist":
        image_size = 28
        linear_in_feautres = image_size * image_size * 1
        channels = 1
    elif args.dataset == "SVHN":
        image_size = 32
        linear_in_feautres = image_size * image_size * 3
        channels = 3
    elif args.dataset == "cifar10":
        image_size = 32
        linear_in_feautres = image_size * image_size * 3
        channels = 3
    elif args.dataset == "cifar100":
        image_size = 32
        linear_in_feautres = image_size * image_size * 3
        channels = 3
    elif args.dataset == "Tiny-ImageNet-200":
        image_size = 64
        linear_in_feautres = image_size * image_size * 3
        channels = 3
    else:
        pass
    return image_size, linear_in_feautres, channels


def get_model(args, num_of_classes=10):
    linear_in_feautres = None
    dataset = args.dataset
    split_config = Split_Configs[args.model][args.split_local_module_num]
    # split_measure_config = Split_Configs[args.model][args.split_measure_local_module_num]
    split_measure_config = EXNN_Split_Configs[args.model][args.split_measure_local_module_num]

    layers = None
    image_size, linear_in_feautres, channels = get_data_info(args)
    if args.type == "fed-expandable":
        small_layers = None
        large_layers = None
        if args.model == "mlp3":
            hidden_features = args.mlp_hidden_features
            layers = mlp3(linear_in_feautres, hidden_features, num_of_classes, init_classifier=False)
            if args.fedexnn_hetero_layer_depth:
                small_layers = mlp3(linear_in_feautres, hidden_features // 2, num_of_classes, init_classifier=False)
                large_layers = mlp3(linear_in_feautres, int(hidden_features * 1.5), num_of_classes, init_classifier=False)
        elif args.model == "cnn":
            hidden_features = args.cnn_hidden_features
            layers = make_CNNCifar_seqs(3, hidden_features, num_of_classes, init_classifier=False)
        elif args.model == "resnet18":
            split_local_layers = fl_exnn_resnet18(group_norm=args.res_group_norm,
                                            res_base_width=args.res_base_width, in_channels=channels, 
                                            hetero_layer_depth=args.fedexnn_hetero_layer_depth)
            layers, small_layers, large_layers = split_local_layers
        elif args.model == "resnet50":
            split_local_layers = fl_exnn_resnet50(group_norm=args.res_group_norm,
                                            res_base_width=args.res_base_width, in_channels=channels, 
                                            hetero_layer_depth=args.fedexnn_hetero_layer_depth)
            layers, small_layers, large_layers = split_local_layers
        else:
            raise NotImplementedError

        # split_config = Split_Configs[args.model][args.fedexnn_split_num]

        split_config = EXNN_Split_Configs[args.model][args.fedexnn_split_num]

        begin_index = 0
        split_modules = []
        for layer_index in split_config:
            split_module = Sequential_SplitNN(None, None, 
                                None, None,
                                layers[begin_index: layer_index+1])
            begin_index = layer_index + 1
            split_modules.append(split_module)
        split_module = Sequential_SplitNN(None, None, 
                            None, None,
                            layers[begin_index:])
        split_modules.append(split_module)
        assert len(split_modules) == args.fedexnn_split_num

        return layers, split_modules


    if args.type == "progressive":
        # if args.model == "pnn":
        if args.model == "mlp3":
            hidden_features = args.mlp_hidden_features
            model = PNN(num_layers=args.num_layers,
                            in_features=linear_in_feautres,
                            hidden_features_per_column=hidden_features,
                            num_of_classes=num_of_classes)
        # elif args.model == "pnn-cnn":
        elif args.model == "cnn":
            hidden_features = args.cnn_hidden_features
            model = PNN_CNN(num_layers=args.num_layers,
                        in_features=channels,
                        hidden_features_per_column=hidden_features,
                        num_of_classes=num_of_classes,
                        adapter="cnn",
                        )
        elif args.model == "resnet18":
            model = pnn_resnet18(num_classes=num_of_classes, group_norm=args.res_group_norm,
                                res_base_width=args.res_base_width, in_channels=channels, adapter="cnn")
        elif args.model == "resnet50":
            model = pnn_resnet50(num_classes=num_of_classes, group_norm=args.res_group_norm,
                                res_base_width=args.res_base_width, in_channels=channels, adapter="cnn")
        return model


    if args.type == "fed-progressive":
        # if args.model == "fl-pnn":
        if args.model == "mlp3":
            hidden_features = args.mlp_hidden_features
            model = Federated_PNN(num_layers=args.num_layers,
                            in_features=3,
                            hidden_features_per_column=hidden_features,
                            num_of_classes=num_of_classes,
                            classifier_name=args.progressive_classifer
                            )
        # elif args.model == "fl-pnn-cnn":
        elif args.model == "cnn":
            hidden_features = args.cnn_hidden_features
            model = Federated_PNN_CNN(num_layers=args.num_layers,
                        in_features=channels,
                        hidden_features_per_column=hidden_features,
                        num_of_classes=num_of_classes,
                        adapter="cnn",
                        classifier_name=args.progressive_classifer
                    )
        elif args.model == "resnet18":
            model = fl_pnn_resnet18(num_classes=num_of_classes, group_norm=args.res_group_norm,
                                res_base_width=args.res_base_width, in_channels=channels,
                                adapter="cnn", classifier_name=args.progressive_classifer)
        elif args.model == "resnet50":
            model = fl_pnn_resnet50(num_classes=num_of_classes, group_norm=args.res_group_norm,
                                res_base_width=args.res_base_width, in_channels=channels,
                                adapter="cnn", classifier_name=args.progressive_classifer)
        return model

    if args.model == "mnist_cnn":
        model = CNNMnist()
    elif args.model == "fmnist_cnn":
        model = CNNMnist()
    elif args.model == "cnn":
        hidden_features = args.cnn_hidden_features
        # model = CNNCifar(hidden_features, num_of_classes)
        layers = make_CNNCifar_seqs(3, hidden_features, num_of_classes, init_classifier=True)
        model = Sequential_SplitNN(args.split_train, split_config, 
                            split_measure_config, args.split_local_module_num,
                            layers)
    elif args.model == "mlp2":
        hidden_features = args.mlp_hidden_features
        layers = mlp2(linear_in_feautres, hidden_features, num_of_classes, init_classifier=True)
        model = Sequential_SplitNN(args.split_train, split_config, 
                            split_measure_config, args.split_local_module_num,
                            layers)
    elif args.model == "mlp3":
        hidden_features = args.mlp_hidden_features
        layers = mlp3(linear_in_feautres, hidden_features, num_of_classes, init_classifier=True)
        model = Sequential_SplitNN(args.split_train, split_config, 
                            split_measure_config, args.split_local_module_num,
                            layers)

    elif args.model == "svhn_cnn":
        hidden_features = args.cnn_hidden_features
        model = CNNCifar(hidden_features, num_of_classes)
    elif args.model == "cifar100_cnn":
        model = CNNCifar100()
    elif args.model == "resnet18":
        # model = resnet18(num_classes=num_of_classes, group_norm=args.res_group_norm, res_base_width=args.res_base_width, in_channels=channels)
        layers = resnet18_layers(init_classifier=True,
            num_classes=num_of_classes, group_norm=args.res_group_norm, res_base_width=args.res_base_width, in_channels=channels)
        model = Sequential_SplitNN(args.split_train, split_config, 
                            split_measure_config, args.split_local_module_num,
                            layers)
        # resnet18_head, resnet50_head
    elif args.model == "resnet50":
        layers = resnet50_layers(init_classifier=True,
            num_classes=num_of_classes, group_norm=args.res_group_norm, res_base_width=args.res_base_width, in_channels=channels)
        model = Sequential_SplitNN(args.split_train, split_config, 
                            split_measure_config, args.split_local_module_num,
                            layers)

    elif args.model == "vit":
        model = deit_tiny_patch16_224(num_classes=num_of_classes,
                                             drop_rate=0.,
                                             drop_path_rate=0.1)
        model.head = torch.nn.Linear(model.head.in_features, num_of_classes)
        model = torch.nn.DataParallel(model)

    return layers, model


def adjust_learning_rate(optimizer, epoch, training_configurations, args):
    """Sets the learning rate"""
    if not args.MI_cos_lr:
        if epoch in training_configurations[args.model]['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

    else:
        for param_group in optimizer.param_groups:
            if epoch <= 10:
                param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs'])) * (epoch - 1) / 10 + 0.01 * (11 - epoch) / 10
            else:
                param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                    * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs']))
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])




def measure_feautre(device, data_loader, model):
    """Eval for one epoch on the training set"""
    model.eval()
    layer_channel_norms = {}
    layer_total_norm = {}

    total_batches = 0
    with torch.no_grad():
        for i, (x, target) in enumerate(data_loader):
            target = target.to(device)
            x = x.to(device)
            output, hidden_xs = model.forward_measure(x)
            if args.type == "fed-expandable":
                hidden_xs = to_exnn_hidden_xs(hidden_xs)
            for layer_idx, features in hidden_xs.items():
                # norm on height and weight, output shape is [batch_size, num_channels]
                norms = torch.norm(features, p=2, dim=[2, 3])

                # average for mini-batch
                # shape is [num_channels]
                batch_mean_norms = torch.mean(norms, dim=0)
                if layer_idx not in layer_channel_norms:
                    layer_channel_norms[layer_idx] = batch_mean_norms
                else:
                    layer_channel_norms[layer_idx] += batch_mean_norms
            total_batches += 1

    for layer_idx in layer_channel_norms.keys():
        layer_channel_norms[layer_idx] = (layer_channel_norms[layer_idx] / total_batches)
        layer_total_norm[layer_idx] = torch.norm(layer_channel_norms[layer_idx], p=2).item()
    return layer_channel_norms, layer_total_norm


def get_all_feature(device, data_loader, model, num_points=1000):
    model.eval()

    layer_feats = {}
    labels = []
    loaded_num_points = 0

    with torch.no_grad():
        for i, (x, target) in enumerate(data_loader):
            x = x.to(device)
            loaded_num_points += x.shape[0]
            output, hidden_xs = model.forward_measure(x)
            if args.type == "fed-expandable":
                hidden_xs = to_exnn_hidden_xs(hidden_xs)
            labels.append(target)
            for layer_idx, features in hidden_xs.items():
                if layer_idx not in layer_feats:
                    layer_feats[layer_idx] = []
                layer_feats[layer_idx].append(features)
            if loaded_num_points > num_points:
                break
        for layer_idx in layer_feats.keys():
            layer_feats[layer_idx] = torch.cat(layer_feats[layer_idx], dim=0)[:num_points].to('cpu')
        labels = torch.cat(labels, dim=0)[:num_points]

    return layer_feats, labels






def estMI(device, train_loader, model, estimator, optimizer, epoch, num_layers):
    """Train for one epoch on the training set"""
    layer_top1s = [AverageMeter() for _ in range(num_layers)]

    record_file = ExpTool.get_file_name("EstiMI.txt", exp_dir=True)
    model.eval()

    loss_ixx_modules_iters = []
    loss_ixy_modules_iters = []

    local_iters = len(train_loader)

    for i, (x, target) in enumerate(train_loader):
        target = target.to(device)
        x = x.to(device)

        optimizer.zero_grad()
        output, hidden_xs = model.forward_measure(x)
        if args.type == "fed-expandable":
            hidden_xs = to_exnn_hidden_xs(hidden_xs)

        # show_model_layers(model, logger=None)
        # for k, decode in decoders.items():
        #     logger.info(f"====decoder {k}==============================")
        #     show_model_layers(decode, logger)
        #     logger.info(f"====aux_classifier {k}==============================")
        #     show_model_layers(aux_classifiers[k], logger)

        # for layer_index, hidden_x in hidden_xs.items():
        #     logging.info(f"layer: {layer_index}, has tensor shape: {hidden_x.shape}")

        h_logits, loss_ixx_modules, loss_ixy_modules = estimator(x, hidden_xs, target)

        loss_ixx_modules_iters.append(loss_ixx_modules)
        loss_ixy_modules_iters.append(loss_ixy_modules)
        optimizer.step()

        for layer_i, logits in enumerate(h_logits):
            prec1 = accuracy(logits.data, target, topk=(1,))[0]
            layer_top1s[layer_i].update(prec1.item(), x.size(0))

        if (i+1) % 10 == 0:
            # print(discriminate_weights)
            fd = open(record_file, 'a+')
            string = f"Training Epoch: [{epoch}][{i}/{local_iters}], loss_ixx: {[round(loss_ixx, 3) for loss_ixx in loss_ixx_modules]} " + \
                f"loss_ixy: {[round(loss_ixy, 3) for loss_ixy in loss_ixy_modules]} " + \
                f"top1s: {[round(top1s.val, 3) for top1s in layer_top1s]} "

            logging.info(string)
            # print(weights)
            fd.write(string + '\n')
            fd.close()

    loss_ixx_modules_iters = np.array(loss_ixx_modules_iters)
    loss_ixy_modules_iters = np.array(loss_ixy_modules_iters)
    loss_ixx_modules_iters = np.mean(loss_ixx_modules_iters, axis=0)
    loss_ixy_modules_iters = np.mean(loss_ixy_modules_iters, axis=0)
    fd = open(record_file, 'a+')
    string = f"Training Epoch: [{epoch}], loss_ixx avg: {[round(loss_ixx, 3) for loss_ixx in loss_ixx_modules_iters]} " + \
            f"loss_ixy avg: {[round(loss_ixy, 3) for loss_ixy in loss_ixy_modules_iters]} " + \
            f"top1s avg: {[round(top1s.avg, 3) for top1s in layer_top1s]} "
    logging.info(string)
    fd.write(string + '\n')
    fd.close()
    loss_ixxs = [round(loss_ixx, 3) for loss_ixx in loss_ixx_modules_iters]
    top1s_avg = [round(top1s.avg, 3) for top1s in layer_top1s]

    return loss_ixxs, top1s_avg



def train_linear_probe(device, train_loader, model, linear_probes, optimizer, epoch, num_layers):
    """Train for one epoch on the training set"""
    layer_top1s = [AverageMeter() for _ in range(num_layers)]

    record_file = ExpTool.get_file_name("EstiMI.txt", exp_dir=True)
    model.eval()

    loss_ixys_iters = []
    local_iters = len(train_loader)
    for i, (x, target) in enumerate(train_loader):
        target = target.to(device)
        x = x.to(device)

        optimizer.zero_grad()
        output, hidden_xs = model.forward_measure(x)
        if args.type == "fed-expandable":
            hidden_xs = to_exnn_hidden_xs(hidden_xs)
        h_logits, loss_ixys = linear_probes(x, hidden_xs, target)
        loss_ixys_iters.append(loss_ixys)
        optimizer.step()

        for layer_i, logits in enumerate(h_logits):
            prec1 = accuracy(logits.data, target, topk=(1,))[0]
            layer_top1s[layer_i].update(prec1.item(), x.size(0))

        if (i+1) % 10 == 0:
            # print(discriminate_weights)
            fd = open(record_file, 'a+')
            string = f"Training Epoch: [{epoch}][{i}/{local_iters}], " + \
                f"loss_ixy: {[round(loss_ixy, 3) for loss_ixy in loss_ixys]} " + \
                f"top1s: {[round(top1s.val, 3) for top1s in layer_top1s]} "

            logging.info(string)
            # print(weights)
            fd.write(string + '\n')
            fd.close()
    loss_ixys_iters = np.array(loss_ixys_iters)
    loss_ixys_iters = np.mean(loss_ixys_iters, axis=0)
    fd = open(record_file, 'a+')
    string = f"Training Epoch: [{epoch}]," + \
            f"loss_ixy avg: {[round(loss_ixy, 3) for loss_ixy in loss_ixys_iters]} " + \
            f"top1s avg: {[round(top1s.avg, 3) for top1s in layer_top1s]} "
    logging.info(string)
    fd.write(string + '\n')
    fd.close()
    top1s_avg = [round(top1s.avg, 3) for top1s in layer_top1s]

    return top1s_avg



def get_res_MIEstimator(split_measure_config, num_of_classes, group_norm, res_base_width, channels):
    layers = resnet18_layers(init_classifier=True,
        num_classes=num_of_classes, group_norm=group_norm, res_base_width=res_base_width, in_channels=channels)

    decoders, aux_classifiers = make_ResNetMIEstimator(
        layers, hidden_x_channels, image_size, aux_net_widen=1)

    mi_estimator = ReconMIEstimator(split_measure_config)

    for layer_index, decoder in decoders.items(): 
        mi_estimator.add_decoder(decoder, layer_index)
    for layer_index, aux_classifier in aux_classifiers.items(): 
        mi_estimator.add_aux_classifier(aux_classifier, layer_index)
    return mi_estimator



if __name__ == '__main__':

    args = args_parser()

    if args.main_task == "train":
        ExpTool.init(args)
    elif args.main_task in ["MI", "LinearProbe"]:
        if not args.exp_tool_init_sub_dir == "no":
            ExpTool.init_with_sub_dir(args, args.exp_tool_init_sub_dir)
        else:
            ExpTool.init(args)
    else:
        raise NotImplementedError

    logger = logging_config(args, 0)
    # wandb.init(config=args,
    #            project="ont-shot FL")

    device = torch.device(f"cuda:{args.gpu}")
    setup_seed(args.seed)
    # pdb.set_trace()
    image_size = get_image_size(args.dataset)
    num_of_classes = get_num_of_labels(args.dataset)
    train_dataset, test_dataset, train_user_groups, train_data_cls_counts, test_user_groups, test_data_cls_counts = partition_data(
        image_size, args.dataset, args.datadir, args.partition, alpha=args.alpha, num_users=args.num_users,
        contrastive_train=args.contrastive_train, contrastive_n_views=args.contrastive_n_views)

    logger.info(f"train_data_cls_counts: {train_data_cls_counts}")
    logger.info(f"test_data_cls_counts: {test_data_cls_counts}")

    global_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                              shuffle=False, num_workers=4)
    # BUILD MODEL

    mi_estimator_configurations = {
        'resnet18': {
            'epochs': 160,
            'batch_size': 128,
            'initial_learning_rate': 0.01,
            # 'batch_size': 1024 if args.dataset in ['cifar10', 'svhn'] else 128,
            # 'initial_learning_rate': 0.8 if args.dataset in ['cifar10', 'svhn'] else 0.1,
            'changing_lr': [80, 120],
            'lr_decay_rate': 0.1,
            'momentum': 0.9,
            'nesterov': True,
            'weight_decay': 1e-4,
        },
        'resnet50': {
            'epochs': 160,
            'batch_size': 1024 if args.dataset in ['cifar10', 'svhn'] else 128,
            'initial_learning_rate': 0.8 if args.dataset in ['cifar10', 'svhn'] else 0.1,
            'changing_lr': [80, 120],
            'lr_decay_rate': 0.1,
            'momentum': 0.9,
            'nesterov': True,
            'weight_decay': 1e-4,
        },
    }

    linear_probe_configurations = {
        'resnet18': {
            'epochs': 10,
            'batch_size': 128,
            'initial_learning_rate': 0.01,
            'momentum': 0.9,
            'nesterov': True,
            'weight_decay': 1e-4,
        },
    }

    layers, global_model = get_model(args, num_of_classes)
    split_measure_config = EXNN_Split_Configs[args.model][args.split_measure_local_module_num]
    # split_measure_config = Split_Configs[args.model][args.split_measure_local_module_num]
    image_size, linear_in_feautres, channels = get_data_info(args)

    if args.model == "resnet18":
        out_channels = get_res18_out_channels(args.res_base_width)
    elif args.model == "mlp2":
        out_channels = [args.mlp_hidden_features for _ in range(2)]
    elif args.model == "mlp3":
        out_channels = [args.mlp_hidden_features for _ in range(3)]
    elif args.model == "cnn":
        pass
    else:
        raise NotImplementedError


    if args.main_task == "train":
        if args.type == "pretrain":
            global_model, global_weights, local_weights, model_list = pretrain(
                args, device, logger, train_dataset, test_dataset, 
                train_user_groups, train_data_cls_counts, 
                test_user_groups, test_data_cls_counts,
                global_test_loader, global_model, out_channels)
        elif args.type == "progressive":
            progressive(args, device, logger, train_dataset, test_dataset, 
            train_user_groups, train_data_cls_counts, 
            test_user_groups, test_data_cls_counts,
            global_test_loader, global_model, out_channels)
        elif args.type == "fed-progressive":
            fed_progressive(args, device, logger, train_dataset, test_dataset, 
            train_user_groups, train_data_cls_counts, 
            test_user_groups, test_data_cls_counts,
            global_test_loader, global_model, out_channels)
        elif args.type == "fed-expandable":
            fed_expandable(args, device, logger, train_dataset, test_dataset, 
            train_user_groups, train_data_cls_counts, 
            test_user_groups, test_data_cls_counts,
            global_test_loader, global_model, out_channels)
        else:
            raise RuntimeError

    elif args.main_task == "MI":
        if args.type == "pretrain":
            assert args.resume
            local_weights = ExpTool.load_pickle(args.resume, exp_dir=False)
            model_list = []
            for i in range(len(local_weights)):
                net = copy.deepcopy(global_model)
                net.load_state_dict(local_weights[i])
                model_list.append(net)
            ensemble_model = Ensemble(model_list)
            # global_model_test_acc, test_loss = test(global_model, global_test_loader, device)
            # logger.info(f"global_model acc: {global_model_test_acc}")

            local_model = model_list[0]
            # local_model_test_acc, test_loss = test(local_model, global_test_loader, device)
            # logger.info(f"local_model acc: {local_model_test_acc}")

            # ensemble_acc, ensemble_loss = test(ensemble_model, global_test_loader, device)
            # logger.info(f"ensemble acc: {ensemble_acc}")
            measure_model = local_model
            if not args.EstFeatNorm == "no":
                idx = 0
                local_train_loader = DataLoader(DatasetSplit(train_dataset, train_user_groups[idx]),
                                            batch_size=args.local_bs, shuffle=True, num_workers=4, drop_last=False)
                local_test_loader = DataLoader(DatasetSplit(test_dataset, test_user_groups[idx]),
                                            batch_size=args.local_bs, shuffle=False, num_workers=4, drop_last=False)
                EstFeatNorm_results = {}
                record_file = ExpTool.get_file_name("EstFeatNorm.txt", exp_dir=True)
                fd = open(record_file, 'a+')
                for client_idx, model in enumerate(model_list):
                    EstFeatNorm_results[client_idx] = {}
                    model.to(device)
                    layer_channel_norms, layer_total_norm = measure_feautre(device, local_train_loader, model)
                    model.to("cpu")
                    EstFeatNorm_results[client_idx]["layer_channel_norms"] = layer_channel_norms
                    EstFeatNorm_results[client_idx]["layer_total_norm"] = layer_total_norm
                    for layer_idx, channel_norms in layer_channel_norms.items():
                        ExpTool.logging_write(f"client_idx:{client_idx}, layer_idx:{layer_idx}: layer_total_norm = {layer_total_norm[layer_idx]}", fd)
                        # ExpTool.logging_write(f"channel_norms:{channel_norms} =============", fd)
                fd.close()
                ExpTool.save_pickle(EstFeatNorm_results, "EstFeatNorm_results", exp_dir=True)
                ExpTool.finish(args)
                exit()

            if not args.SaveFeats == "no":
                local_FeatLabels_results = {}
                for client_idx, model in enumerate(model_list):
                    if client_idx > 1:
                        break
                    logging.info(f"get client {client_idx} features")
                    model.to(device)
                    layer_feats, labels = get_all_feature(device, global_test_loader, model, num_points=1000)
                    model.to("cpu")
                    local_FeatLabels_results[client_idx] = {
                        "layer_feats": layer_feats,
                        "labels": labels}
                ExpTool.save_pickle(local_FeatLabels_results, "local_FeatLabels_results", exp_dir=True)
                # global_FeatLabels_results = {}
                # for client_idx, model in enumerate(model_list):
                #     logging.info(f"get client {client_idx} features")
                #     model.to(device)
                #     layer_feats, labels = get_all_feature(device, global_test_loader, model, num_points=1000)
                #     model.to("cpu")
                #     global_FeatLabels_results[client_idx] = {
                #         "layer_feats": layer_feats,
                #         "labels": labels}
                # ExpTool.save_pickle(global_FeatLabels_results, "global_FeatLabels_results", exp_dir=True)
                # if not args.TSNE == "no":
                ExpTool.load_pickle("local_FeatLabels_results", exp_dir=True)
                avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                # avg_pool.to(device)
                for client_idx in local_FeatLabels_results.keys():
                    layer_feats = local_FeatLabels_results[client_idx]["layer_feats"]
                    labels = local_FeatLabels_results[client_idx]["labels"]
                    for layer_index, features in layer_feats.items():
                        logging.info(f"T-SNE on client {client_idx}, layer {layer_index} ...... ")
                        tSNE_save_path = ExpTool.get_file_name(f"local_c{client_idx}_l{layer_index}_TSNE.pdf", exp_dir=True)
                        if len(features.shape) > 2:
                            features = avg_pool(features[:args.TSNE_points])
                        features = features.view(features.shape[0], -1)
                        draw_tsne(device, num_of_classes, features, labels[:args.TSNE_points],
                            tSNE_save_path=tSNE_save_path)
                # ExpTool.load_pickle("global_FeatLabels_results", exp_dir=True)
                # for client_idx in global_FeatLabels_results.keys():
                #     layer_feats = global_FeatLabels_results[client_idx]["layer_feats"]
                #     labels = global_FeatLabels_results[client_idx]["labels"]
                #     for layer_index, features in layer_feats.items():
                #         tSNE_save_path = ExpTool.get_file_name(f"global_c{client_idx}_l{layer_index}_TSNE.pdf", exp_dir=True)
                #         draw_tsne(device, num_of_classes, layer_feats, labels,
                #             tSNE_save_path=tSNE_save_path)
                ExpTool.finish(args)
                exit()


        elif args.type == "fed-expandable":
            assert args.resume
            global_model = init_fedexnn_merged(args, global_model, out_channels)
            weights = ExpTool.load_pickle(args.resume, exp_dir=False)
            # show_model_layers(global_model, logger=None)
            # logger.info(f"================================")
            # for k, v in weights.items():
            #     logger.info(f"layer: {k}, Shape:{v.shape} No. Params: {v.numel()}")
            global_model.load_state_dict(weights)
            # global_model.load(weights)
            measure_model = global_model
        else:
            raise RuntimeError
        in_channels = []
        measure_model.eval()
        measure_model.to(device)

        for i, (x, target) in enumerate(global_test_loader):
            target = target.to(device)
            x = x.to(device)
            output, hidden_xs = measure_model.forward_measure(x)
            break

        def to_exnn_hidden_xs(hidden_xs):
            if args.type == "fed-expandable":
                # map to normal layer index
                split_config = EXNN_Split_Configs[args.model][args.fedexnn_split_num]
                new_hidden_xs = {}
                for module_idx, layer_idx in enumerate(split_config):
                    new_hidden_xs[layer_idx] = hidden_xs[module_idx]
            return new_hidden_xs

        if args.type == "fed-expandable":
            hidden_xs = to_exnn_hidden_xs(hidden_xs)

        hidden_x_channels = dict([(k, h.shape[1]) for k, h in hidden_xs.items()])
        logging.info(f"========== hidden_x_channels: {hidden_x_channels}")

        if args.model in ["resnet18"]:
            mi_estimator = get_res_MIEstimator(split_measure_config, num_of_classes, args.res_group_norm, args.res_base_width, channels)
            global_train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=mi_estimator_configurations[args.model]['batch_size'],
                shuffle=False, num_workers=4)
            num_layers = args.split_measure_local_module_num

            optimizer = torch.optim.SGD(
                mi_estimator.parameters(),
                lr=mi_estimator_configurations[args.model]['initial_learning_rate'],
                momentum=mi_estimator_configurations[args.model]['momentum'],
                nesterov=mi_estimator_configurations[args.model]['nesterov'],
                weight_decay=mi_estimator_configurations[args.model]['weight_decay'])

            # show_model_layers(global_model, logger)
            # for k, decode in decoders.items():
            #     logger.info(f"====decoder {k}==============================")
            #     show_model_layers(decode, logger)
            #     logger.info(f"====aux_classifier {k}==============================")
            #     show_model_layers(aux_classifiers[k], logger)


            mi_estimator.to(device)
            MI_results = {}

            for epoch in range(0, mi_estimator_configurations[args.model]['epochs']):
                # adjust_learning_rate(optimizer, epoch + 1)
                if args.debug and epoch == 1:
                    break
                adjust_learning_rate(optimizer, epoch, mi_estimator_configurations, args)
                train_loss_ixxs, train_top1s_avg = estMI(device, global_train_loader, measure_model, mi_estimator, optimizer, epoch, num_layers)
                if epoch % 10 == 0 or epoch == mi_estimator_configurations[args.model]['epochs'] - 1:
                    MI_results[epoch] = {}
                    loss_ixx_modules_iters = []
                    loss_ixy_modules_iters = []
                    layer_test_top1s = [AverageMeter() for _ in range(len(hidden_xs))]
                    for i, (x, target) in enumerate(global_test_loader):
                        target = target.to(device)
                        x = x.to(device)
                        output, hidden_xs = measure_model.forward_measure(x)
                        if args.type == "fed-expandable":
                            hidden_xs = to_exnn_hidden_xs(hidden_xs)
                        h_logits, loss_ixx_modules, loss_ixy_modules = mi_estimator(x, hidden_xs, target)
                        loss_ixx_modules_iters.append(loss_ixx_modules)
                        loss_ixy_modules_iters.append(loss_ixy_modules)
                        for layer_i, logits in enumerate(h_logits):
                            prec1 = accuracy(logits.data, target, topk=(1,))[0]
                            layer_test_top1s[layer_i].update(prec1.item(), x.size(0))

                    record_file = ExpTool.get_file_name("EstiMI.txt", exp_dir=True)
                    loss_ixx_modules_iters = np.array(loss_ixx_modules_iters)
                    loss_ixy_modules_iters = np.array(loss_ixy_modules_iters)
                    loss_ixx_modules_iters = np.mean(loss_ixx_modules_iters, axis=0)
                    loss_ixy_modules_iters = np.mean(loss_ixy_modules_iters, axis=0)
                    fd = open(record_file, 'a+')

                    string = f"Testing Epoch: [{epoch}], loss_ixx avg: {[round(loss_ixx, 3) for loss_ixx in loss_ixx_modules_iters]} " + \
                            f"loss_ixy avg: {[round(loss_ixy, 3) for loss_ixy in loss_ixy_modules_iters]} " + \
                            f"top1s avg: {[round(top1s.avg, 3) for top1s in layer_test_top1s]} "
                    test_loss_ixxs = [round(loss_ixx, 3) for loss_ixx in loss_ixx_modules_iters]
                    test_top1s_avg = [round(top1s.avg, 3) for top1s in layer_test_top1s]
                    print(string)
                    fd.write(string + '\n')
                    fd.close()
                    MI_results[epoch]["train_loss_ixxs"] = train_loss_ixxs
                    MI_results[epoch]["train_top1s_avg"] = train_top1s_avg
                    MI_results[epoch]["test_loss_ixxs"] = test_loss_ixxs
                    MI_results[epoch]["test_top1s_avg"] = test_top1s_avg
            ExpTool.save_pickle(MI_results, "MI_results", exp_dir=True)
        else:
            raise NotImplementedError

        ExpTool.finish(args)
    elif args.main_task == "LinearProbe":
        if args.type == "pretrain":
            assert args.resume
            local_weights = ExpTool.load_pickle(args.resume, exp_dir=False)
            model_list = []
            for i in range(len(local_weights)):
                net = copy.deepcopy(global_model)
                net.load_state_dict(local_weights[i])
                model_list.append(net)
            ensemble_model = Ensemble(model_list)
            local_model = model_list[0]
            measure_model = local_model
        elif args.type == "fed-expandable":
            assert args.resume
            global_model = init_fedexnn_merged(args, global_model, out_channels)
            weights = ExpTool.load_pickle(args.resume, exp_dir=False)
            global_model.load_state_dict(weights)
            measure_model = global_model
        else:
            raise RuntimeError
        in_channels = []
        measure_model.eval()
        measure_model.to(device)

        for i, (x, target) in enumerate(global_test_loader):
            target = target.to(device)
            x = x.to(device)
            output, hidden_xs = measure_model.forward_measure(x)
            break

        def to_exnn_hidden_xs(hidden_xs):
            if args.type == "fed-expandable":
                # map to normal layer index
                split_config = EXNN_Split_Configs[args.model][args.fedexnn_split_num]
                new_hidden_xs = {}
                for module_idx, layer_idx in enumerate(split_config):
                    new_hidden_xs[layer_idx] = hidden_xs[module_idx]
            return new_hidden_xs

        if args.type == "fed-expandable":
            hidden_xs = to_exnn_hidden_xs(hidden_xs)

        hidden_x_channels = dict([(k, h.shape[1]) for k, h in hidden_xs.items()])
        logging.info(f"========== hidden_x_channels: {hidden_x_channels}")
        if args.model in ["resnet18"]:
            linear_probes = LinearProbes()
            for layer_index, h in hidden_xs.items():
                linear_probes.add(layer_index, h[0].numel(), num_of_classes)
            global_train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=linear_probe_configurations[args.model]['batch_size'],
                shuffle=False, num_workers=4)
            num_layers = args.split_measure_local_module_num

            optimizer = torch.optim.SGD(
                linear_probes.parameters(),
                lr=linear_probe_configurations[args.model]['initial_learning_rate'],
                momentum=linear_probe_configurations[args.model]['momentum'],
                nesterov=linear_probe_configurations[args.model]['nesterov'],
                weight_decay=linear_probe_configurations[args.model]['weight_decay'])
            linear_probes.to(device)
            linear_probe_results = {}
            for epoch in range(0, linear_probe_configurations[args.model]['epochs']):
                # adjust_learning_rate(optimizer, epoch + 1)
                if args.debug and epoch == 1:
                    break
                top1s_avg = train_linear_probe(device, global_train_loader, measure_model, linear_probes, optimizer, epoch, num_layers)
                if epoch % 10 == 0 or epoch == linear_probe_configurations[args.model]['epochs'] - 1:
                    linear_probe_results[epoch] = {}
                    loss_ixy_modules_iters = []
                    layer_test_top1s = [AverageMeter() for _ in range(len(hidden_xs))]
                    for i, (x, target) in enumerate(global_test_loader):
                        target = target.to(device)
                        x = x.to(device)
                        output, hidden_xs = measure_model.forward_measure(x)
                        if args.type == "fed-expandable":
                            hidden_xs = to_exnn_hidden_xs(hidden_xs)
                        h_logits, loss_ixy_modules = linear_probes(x, hidden_xs, target)
                        loss_ixy_modules_iters.append(loss_ixy_modules)
                        for layer_i, logits in enumerate(h_logits):
                            prec1 = accuracy(logits.data, target, topk=(1,))[0]
                            layer_test_top1s[layer_i].update(prec1.item(), x.size(0))

                    record_file = ExpTool.get_file_name("LinearProbeResults.txt", exp_dir=True)
                    loss_ixy_modules_iters = np.array(loss_ixy_modules_iters)
                    loss_ixy_modules_iters = np.mean(loss_ixy_modules_iters, axis=0)
                    fd = open(record_file, 'a+')

                    string = f"Testing Epoch: [{epoch}], " + \
                            f"loss_ixy avg: {[round(loss_ixy, 3) for loss_ixy in loss_ixy_modules_iters]} " + \
                            f"top1s avg: {[round(top1s.avg, 3) for top1s in layer_test_top1s]} "
                    test_top1s_avg = [round(top1s.avg, 3) for top1s in layer_test_top1s]
                    print(string)
                    fd.write(string + '\n')
                    fd.close()
                    linear_probe_results[epoch]["test_top1s_avg"] = test_top1s_avg

    else:
        raise NotImplementedError


































