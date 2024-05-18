#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import copy
import os
import shutil
import sys
import warnings
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import pdb
import logging
import time


from helpers.datasets import partition_data, load_data, get_image_size, get_num_of_labels
from helpers.utils import get_dataset, average_weights, DatasetSplit, BackdoorDS, KLDiv, setup_seed, test, progressive_test
from helpers.exp_path import ExpTool


from models.generator import Generator
from models.nets import CNNCifar, CNNMnist, CNNCifar100
from models.pnn import PNN
from models.pnn_cnn import PNN_CNN, pnn_resnet18, pnn_resnet50

from models.fl_pnn import Federated_PNN
from models.fl_pnn_cnn import Federated_PNN_CNN, fl_pnn_resnet18, fl_pnn_resnet50
from models.mlp import MLP
from models.fl_exnn import (MLP_Block, CNN_Block,
    merge_layer, Federated_EXNN, Federated_EXNNLayer_global, Federated_EXNNLayer_local,
    fl_exnn_resnet18, fl_exnn_resnet50, 
)
from models.seq_model import Sequential_SplitNN, ReconMIEstimator


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from models.resnet import resnet18, resnet50, get_res18_out_channels
from models.vit import deit_tiny_patch16_224
import wandb
from models.configs import Split_Configs, EXNN_Split_Configs

warnings.filterwarnings('ignore')
upsample = torch.nn.Upsample(mode='nearest', scale_factor=7)

from locals.fedavg import LocalUpdate
from locals.fl_progressive import FedPnnLocalUpdate
from locals.progressive import PnnLocalUpdate
from locals.fl_expandable import FedEXNNLocalUpdate
from locals.ccvr import (compute_classes_mean_cov, generate_virtual_representation,
    calibrate_classifier, get_means_covs_from_client)

from utils import seq_map_values, batch, accuracy, show_model_layers



def obtain_projection_head(before_cls_feature_num, contrastive_projection_dim):
    projector = nn.Sequential(
        nn.Linear(before_cls_feature_num, before_cls_feature_num, bias=False),
        nn.ReLU(),
        nn.Linear(before_cls_feature_num, contrastive_projection_dim, bias=False),
    )
    return projector


class Ensemble(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = model_list

    def to(self, device):
        for model in self.models:
            model.to(device)


    def forward(self, x):
        logits_total = 0
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_total += logits
        logits_e = logits_total / len(self.models)

        return logits_e


def pretrain(args, device, logger, train_dataset, test_dataset, 
            train_user_groups, train_data_cls_counts, 
            test_user_groups, test_data_cls_counts,
            global_test_loader, global_model, out_channels):

    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    users = []
    locals = []

    before_cls_feature_num = out_channels[-1]
    backdoor_test_loader = None
    if args.backdoor_train: 
        backdoor_test_loader = DataLoader(BackdoorDS(test_dataset, args.backdoor_size, mode="random"),
                                    batch_size=256, shuffle=False, num_workers=4)

    # ===============================================
    for idx in range(args.num_users):
        logger.info("client {}".format(idx))
        users.append("client_{}".format(idx))
        if args.backdoor_train and idx < args.backdoor_n_clients:
            local_update = LocalUpdate(args, train_dataset, test_dataset, global_test_loader,
                train_user_groups[idx], test_user_groups[idx], copy.deepcopy(global_model), backdoor_train=True)
        else:
            local_update = LocalUpdate(args, train_dataset, test_dataset, global_test_loader,
                train_user_groups[idx], test_user_groups[idx], copy.deepcopy(global_model))
        locals.append(local_update)
        if args.contrastive_train:
            # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
            projector = obtain_projection_head(before_cls_feature_num, args.contrastive_projection_dim)
            local_update.add_CL_head(projector)

    train_time = 0
    total_epoch = 0
    for epoch in range(args.local_ep):
        start_time = time.time()
        local_weights = []
        train_losses = []
        acc_list = []
        pfl_acc_list = []
        training_pfl_acc_list = []
        if epoch % 10 == 0 or epoch < 10 or epoch == args.local_ep - 1:
            if_test = True
        else:
            if_test = False
        if_test = True
        for idx in range(args.num_users):
            # not load global model, for one-shot communication...
            w, avg_train_loss, global_acc, pfl_acc, train_pfl_acc = locals[idx].update_weights(idx, 1, device, if_test=if_test)
            acc_list.append(global_acc)
            train_losses.append(avg_train_loss)
            pfl_acc_list.append(pfl_acc)
            training_pfl_acc_list.append(train_pfl_acc)
            # local_weights.append(copy.deepcopy(w))
            local_weights.append(w)

        total_epoch += args.local_ep

        avg_train_loss = np.mean(train_losses)
        train_time += time.time() - start_time

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        model_list = []
        for i in range(len(local_weights)):
            net = copy.deepcopy(global_model)
            net.load_state_dict(local_weights[i])
            model_list.append(net)
        ensemble_model = Ensemble(model_list)
        if if_test:
            result_dict = {}
            for idx in range(args.num_users):
                result_dict["client_{}_acc".format(users[idx])] = acc_list[idx]
                result_dict["pfl_acc_on_{}".format(users[idx])] = pfl_acc_list[idx]
                result_dict["pfl_training_acc_on_{}".format(users[idx])] = training_pfl_acc_list[idx]

            ExpTool.record(result_dict)
            test_acc, test_loss = test(global_model, global_test_loader, device)
            logger.info(f"avg acc: {test_acc}")

            ensemble_acc, ensemble_loss = test(ensemble_model, global_test_loader, device)
            if args.backdoor_train:
                ensemble_backdoor_acc, ensemble_backdoor_loss = test(ensemble_model, backdoor_test_loader, device)
                logger.info(f"ensemble_backdoor_acc: {ensemble_backdoor_acc}")
                ExpTool.record({"ensemble_backdoor_acc": ensemble_backdoor_acc,
                                "ensemble_backdoor_loss": ensemble_backdoor_loss})

            logger.info(f"ensemble acc: {ensemble_acc}")
            ExpTool.record({"comm_round": 0, "local_epoch": total_epoch, "train_loss": avg_train_loss,
                            "test_acc": test_acc, "ensemble_acc": ensemble_acc, "train_time": train_time})
            ExpTool.upload()

    count_para = 0
    for local_weight in local_weights:
        # for key, value in local_weight.named_parameters():
        for key, value in local_weight.items():
            count_para += value.numel()
    summary_dict = {"count_paras": count_para}
    logger.info(f"summary_dict: {summary_dict}")
    ExpTool.summary(summary_dict)
    # ===============================================
    if not args.checkpoint == "no":
        ExpTool.save_pickle(local_weights, args.checkpoint, exp_dir=True)
    # ExpTool.load_pickle
    # torch.save(local_weights, '{}_{}clients_{}.pkl'.format(args.dataset, args.num_users, args.alpha))
    return global_model, global_weights, local_weights, model_list






def progressive(args, device, logger, train_dataset, test_dataset, 
            train_user_groups, train_data_cls_counts, 
            test_user_groups, test_data_cls_counts,
            global_test_loader, global_model, out_channels):

    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    users = []
    locals = []

    # ===============================================
    for idx in range(args.num_users):
        logger.info("client {}".format(idx))
        users.append("client_{}".format(idx))
        local_update = PnnLocalUpdate(args, train_dataset, test_dataset, global_test_loader,
                            train_user_groups[idx], test_user_groups[idx])
        locals.append(local_update)

    global_model.train()
    global_model.to(device)
    # Now, there is no local weights in progressive FL, because the model is increasing...
    training_pfl_acc_list = []
    train_losses = []
    train_time = 0
    for idx in range(args.num_users):
        start_time = time.time()
        # not load global model, for one-shot communication...
        _, avg_train_loss, _, train_pfl_acc = locals[idx].update_weights(idx, args.local_ep, global_model, device, if_test=True)
        training_pfl_acc_list.append(train_pfl_acc)
        train_losses.append(avg_train_loss)
        train_time += time.time() - start_time

    avg_train_loss = np.mean(train_losses)

    # Test global and ensemble model
    # NOTE: global weights need not to be averaged
    num_total_corrects = 0
    num_total = 0
    pfl_accs = []
    for idx in range(args.num_users):
        local = locals[idx]
        num_total += len(local.global_test_loader.dataset)
        pfl_acc, pfl_test_loss, correct = progressive_test(global_model, local.global_test_loader, idx, device)
        pfl_accs.append(pfl_acc)
        num_total_corrects += correct
    test_acc = 100. * num_total_corrects / num_total

    result_dict = {}
    for idx in range(args.num_users):
        result_dict["pfl_acc_on_{}".format(users[idx])] = pfl_accs[idx]
        result_dict["pfl_training_acc_on_{}".format(users[idx])] = training_pfl_acc_list[idx]

    logger.info(f"pfl_accs: {pfl_accs}")
    logger.info(f"training_pfl_acc_list:{training_pfl_acc_list}")
    logger.info(f"test_acc:{test_acc}")

    ExpTool.record(result_dict)
    logger.info("avg acc:")
    ExpTool.record({"comm_round": 0, "local_epoch": args.local_ep, "train_loss": avg_train_loss,
                    "test_acc": test_acc, "train_time": train_time})
    ExpTool.upload()

    count_para = 0
    for key, value in global_model.named_parameters():
        count_para += value.numel()
    summary_dict = {"count_paras": count_para}
    logger.info(f"summary_dict: {summary_dict}")
    ExpTool.summary(summary_dict)

    # ===============================================
    if not args.checkpoint == "no":
        ExpTool.save_pickle(global_model.cpu().state_dict(), args.checkpoint, exp_dir=True)
    # torch.save(global_model.cpu().state_dict(), '{}_{}_{}clients_{}.pkl'.format(args.type, args.dataset, args.num_users, args.alpha))


def fed_progressive(args, device, logger, train_dataset, test_dataset, 
            train_user_groups, train_data_cls_counts, 
            test_user_groups, test_data_cls_counts,
            global_test_loader, global_model, out_channels):

    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    users = []
    locals = []
    for idx in range(args.num_users):
        logger.info("client {}".format(idx))
        users.append("client_{}".format(idx))
        local_update = FedPnnLocalUpdate(args, train_dataset, test_dataset, global_test_loader,
                            train_user_groups[idx], test_user_groups[idx])
        locals.append(local_update)

    global_model.train()
    global_model.to(device)
    # Now, there is no local weights in progressive FL, because the model is increasing...
    training_pfl_acc_list = []
    train_losses = []
    train_time = 0
    for idx in range(args.num_users):
        # not load global model, for one-shot communication...
        start_time = time.time()
        _, avg_train_loss, _, train_pfl_acc = locals[idx].update_weights(idx, args.local_ep, global_model, device, if_test=True)
        training_pfl_acc_list.append(train_pfl_acc)
        train_losses.append(avg_train_loss)
        train_time += time.time() - start_time

    avg_train_loss = np.mean(train_losses)

    # Test global and ensemble model
    # NOTE: global weights need not to be averaged
    logger.info("avg acc:")
    test_acc, test_loss = test(global_model, global_test_loader, device)
    pfl_accs = []

    result_dict = {}
    for idx in range(args.num_users):
        result_dict["pfl_training_acc_on_{}".format(users[idx])] = training_pfl_acc_list[idx]
        local_test_acc, _ = test(global_model, locals[idx].test_loader, device)
        result_dict["pfl_acc_on_{}".format(users[idx])] = local_test_acc
        pfl_accs.append(local_test_acc)

    logger.info(f"pfl_accs: {pfl_accs}")
    logger.info(f"training_pfl_acc_list:{training_pfl_acc_list}")
    logger.info(f"test_acc:{test_acc}")

    ExpTool.record(result_dict)
    logger.info("avg acc:")
    ExpTool.record({"comm_round": 0, "local_epoch": args.local_ep, "train_loss": avg_train_loss,
                    "test_acc": test_acc, "train_time": train_time})
    ExpTool.upload()
    count_para = 0
    for key, value in global_model.named_parameters():
        count_para += value.numel()
    summary_dict = {"count_paras": count_para}
    logger.info(f"summary_dict: {summary_dict}")
    ExpTool.summary(summary_dict)

    # ===============================================
    if not args.checkpoint == "no":
        ExpTool.save_pickle(global_model.cpu().state_dict(), args.checkpoint, exp_dir=True)
    # torch.save(global_model.cpu().state_dict(), '{}_{}_{}clients_{}.pkl'.format(args.type, args.dataset, args.num_users, args.alpha))



def init_fedexnn_merged(args, split_modules, out_channels):
    users = []
    local_FedEXNN_models = {}
    split_config = EXNN_Split_Configs[args.model][args.fedexnn_split_num]
    num_of_classes = get_num_of_labels(args.dataset)

    for idx in range(args.num_users):
        split_local_layers = []
        for layer_idx, layer in enumerate(split_modules):
            EXNNLayer_local = Federated_EXNNLayer_local(layer_idx=layer_idx,
                local_layer=copy.deepcopy(layer),
                client_idx=idx,
                adapter=args.fedexnn_adapter,
                fedexnn_self_dropout=args.fedexnn_self_dropout)
            split_local_layers.append(EXNNLayer_local)
        init_model = Federated_EXNN(
            args,
            idx,
            split_local_layers=split_local_layers,
            num_of_classes=num_of_classes,
            fedexnn_classifer=args.fedexnn_classifer)
        local_FedEXNN_models[idx] = init_model

    for idx in range(args.fedexnn_split_num):
        layer_idx = idx
        federated_EXNNLayer_global = merge_layer(local_FedEXNN_models, layer_idx)
        federated_EXNNLayer_global.freeze()
        # split_local_layers[layer_idx] = federated_EXNNLayer_global
        for client_idx in range(args.num_users):
            local_FedEXNN_models[client_idx].adaptation(
                layer_idx, federated_EXNNLayer_global)
            if layer_idx < len(split_config):
                actual_layer_index = split_config[layer_idx]
                local_FedEXNN_models[client_idx].add_local_layer_adaptor(layer_idx+1,
                    in_channels=out_channels[actual_layer_index]*args.num_users,
                    out_channels=out_channels[actual_layer_index])
    return local_FedEXNN_models[0]



def fed_expandable(args, device, logger, train_dataset, test_dataset, 
            train_user_groups, train_data_cls_counts, 
            test_user_groups, test_data_cls_counts,
            global_test_loader, split_modules, out_channels):

    bst_acc = -1
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
    users = []
    locals = []
    split_config = EXNN_Split_Configs[args.model][args.fedexnn_split_num]
    num_of_classes = get_num_of_labels(args.dataset)

    before_cls_feature_num = out_channels[-1]
    backdoor_test_loader = None
    if args.backdoor_train: 
        backdoor_test_loader = DataLoader(BackdoorDS(test_dataset, args.backdoor_size, mode="random"),
                                    batch_size=256, shuffle=False, num_workers=4)

    local_FedEXNN_models = {}
    for idx in range(args.num_users):
        logger.info("client {}".format(idx))
        users.append("client_{}".format(idx))
        # split_local_layers = copy.deepcopy(split_modules)
        split_local_layers = []
        for layer_idx, layer in enumerate(split_modules):
            EXNNLayer_local = Federated_EXNNLayer_local(layer_idx=layer_idx,
                local_layer=copy.deepcopy(layer),
                client_idx=idx,
                adapter=args.fedexnn_adapter,
                fedexnn_self_dropout=args.fedexnn_self_dropout)
            split_local_layers.append(EXNNLayer_local)
        init_model = Federated_EXNN(
            args,
            idx,
            split_local_layers=split_local_layers,
            num_of_classes=num_of_classes,
            fedexnn_classifer=args.fedexnn_classifer)
        if args.debug_show_exnn_id:
            logging.info(f"==========Checking local layer IDs ================")
            logging.info(f"==========Client:{idx}, split_local_layers :{id(split_local_layers)} ================")
            for layer_idx, layer in enumerate(split_local_layers):
                logging.info(f"==========Client:{idx}, layer_idx{layer_idx} :{id(layer)} ================")
            logging.info(f"==========Client:{idx}, init_model :{id(init_model)} ================")

        local_FedEXNN_models[idx] = init_model
        if args.backdoor_train and idx < args.backdoor_n_clients:
            local_update = FedEXNNLocalUpdate(args, train_dataset, test_dataset, global_test_loader,
                                train_user_groups[idx], test_user_groups[idx], backdoor_train=True)
        else:
            local_update = FedEXNNLocalUpdate(args, train_dataset, test_dataset, global_test_loader,
                                train_user_groups[idx], test_user_groups[idx])
        locals.append(local_update)
        if args.contrastive_train:
            # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
            projector = obtain_projection_head(before_cls_feature_num, args.contrastive_projection_dim)
            local_update.add_CL_head(projector)


    # Train and fuse split layers
    count_para = 0
    # if args.debug:
    # show_model_layers(init_model)
    for key, value in init_model.named_parameters():
        count_para += value.numel()
    logger.info(f"init_model has count_para: {count_para}")

    for idx in range(args.fedexnn_split_num):
        pfl_acc_list = []
        training_pfl_acc_list = []
        train_losses = []
        result_dict = {}

        for client_idx in range(args.num_users):
            # not load global model, for one-shot communication...
            _, train_loss, _, pfl_acc, train_pfl_acc = locals[client_idx].update_weights(
                        client_idx, args.local_ep, local_FedEXNN_models[client_idx], device, if_test=True)
            pfl_acc_list.append(pfl_acc)
            training_pfl_acc_list.append(train_pfl_acc)
            train_losses.append(train_loss)
            result_dict["pfl_acc_on_{}".format(users[client_idx])] = pfl_acc
            result_dict["pfl_training_acc_on_{}".format(users[client_idx])] = training_pfl_acc_list[client_idx]
        avg_train_loss = np.mean(train_losses)
        logger.info(f"test_pfl_acc_list:{pfl_acc_list}")
        logger.info(f"training_pfl_acc_list:{training_pfl_acc_list}")
        # logger.info(f"test_acc:{test_acc}")
        # if idx == 0:
        #     pass
        # else:
        layer_idx = idx
        logger.info(f"=====Merging Layer : {layer_idx} =====")
        federated_EXNNLayer_global = merge_layer(local_FedEXNN_models, layer_idx)
        federated_EXNNLayer_global.freeze()
        # split_local_layers[layer_idx] = federated_EXNNLayer_global
        for client_idx in range(args.num_users):
            local_FedEXNN_models[client_idx].adaptation(
                layer_idx, federated_EXNNLayer_global)
            if args.debug_show_exnn_id:
                logging.info(f"==========Checking global layer IDs ================")
                model = local_FedEXNN_models[client_idx]
                logging.info(f"==========Client:{client_idx}, local_FedEXNN_models.layers[{layer_idx}] : \
                            \n ========== {id(model.layers[layer_idx])} ================")
                for sub_client_idx, local_layer, in federated_EXNNLayer_global.local_layers.items():
                    if hasattr(local_layer, "adapter_nn"):
                        logging.info(f"==========In global layer Client:{sub_client_idx},  \
                            \n ================In global layer  local_FedEXNN_models.layers[{layer_idx}].adapter_nn: {id(local_layer.adapter_nn)}")
            if layer_idx < len(split_config):
                actual_layer_index = split_config[layer_idx]
                local_FedEXNN_models[client_idx].add_local_layer_adaptor(layer_idx+1,
                    in_channels=out_channels[actual_layer_index]*args.num_users,
                    out_channels=out_channels[actual_layer_index])
                if args.debug_show_exnn_id:
                    if hasattr(model.layers[layer_idx+1], "adapter_nn"):
                        logging.info(f"==========Client:{client_idx},  \
                            \n ================local_FedEXNN_models.layers[{layer_idx+1}].adapter_nn: {id(model.layers[layer_idx+1].adapter_nn)}")
            if args.debug_show_exnn_id:
                measure_model = local_FedEXNN_models[client_idx]
                try:
                    if getattr(measure_model.layers[0], "is_global", False):
                        logger.info(f'client_idx: {client_idx} layer0 - model weight: {measure_model.layers[0].local_layers["0"].local_layer._layers[0][0].weight.data.norm()}')
                    if getattr(measure_model.layers[1], "is_global", False):
                        logger.info(f'client_idx: {client_idx} layer1 - model (is_global)  has attr adapter_nn : {hasattr(measure_model.layers[1].local_layers[str(client_idx)], "adapter_nn")}')
                        logger.info(f'client_idx: {client_idx} layer1 - model (is_global)  weight: {measure_model.layers[1].local_layers[str(client_idx)].adapter_nn.weight.data.norm()}')
                    else:
                        logger.info(f'client_idx: {client_idx} layer1 - model (isnot_global) has attr adapter_nn : {hasattr(measure_model.layers[1], "adapter_nn")}')
                        if hasattr(measure_model.layers[1], "adapter_nn"):
                            logger.info(f'client_idx: {client_idx} layer1 - model (isnot_global) weight: {measure_model.layers[1].adapter_nn.weight.data.norm()}')
                    if not getattr(measure_model.layers[2], "is_global", False):
                        logger.info(f'client_idx: {client_idx} local layer2 - model weight: {measure_model.layers[2].local_layer._layers[1].conv1.weight.data.norm()}')
                        logger.info(f'client_idx: {client_idx} local layer2 - model weight: {measure_model.layers[2].local_layer._layers[1].conv2.weight.data.norm()}')
                except:
                    pass
        logger.info(f"=====Finish Merging Layer : {layer_idx} =====")
        ExpTool.record(result_dict)
        logger.info(f"result_dict: {result_dict}")
        ExpTool.record({"comm_round": idx, "local_epoch": args.local_ep, "train_loss": avg_train_loss})
        ExpTool.upload()
    if args.debug_show_exnn_id:
        for client_idx in range(args.num_users):
            logging.info(f"==========Checking global layer IDs ================")
            model = local_FedEXNN_models[client_idx]
            for layer_index, layer in enumerate(model.layers):
                logging.info(f"==========Client:{client_idx}, local_FedEXNN_models.layers[{layer_idx}] : \
                                \n ========== {id(layer)} ================")

    for _, model in local_FedEXNN_models.items():
        model.to("cpu")
    global_model = local_FedEXNN_models[0]
    # if args.debug:
    #     show_model_layers(global_model)

    # Train and fuse classifier
    if args.fedexnn_classifer == "avg":
        new_classifier_weights = average_weights([
            local_FedEXNN_model.classifier.cpu().state_dict()  for local_FedEXNN_model in local_FedEXNN_models.values()])
        new_classifier = list(local_FedEXNN_models.values())[0].classifier
        new_classifier.load_state_dict(new_classifier_weights)

    elif args.fedexnn_classifer == "multihead":
        new_classifier = [
            local_FedEXNN_model.classifier  for local_FedEXNN_model in local_FedEXNN_models.values()]
    else:
        raise NotImplementedError

    global_model.adaptation_classifier(fedexnn_classifer=args.fedexnn_classifer, new_classifier=new_classifier)
    # global_model.train()
    global_model.to(device)
    # Now, there is no local weights in progressive FL, because the model is increasing...
    if args.fedexnn_classifer in ["avg"] :
        if args.contrastive_train:
            # Get the normal dataloader without n views.
            image_size = get_image_size(args.dataset)
            _, _, _, _, train_dataset, test_dataset = load_data(
                image_size, args.dataset, args.datadir)
            dataloaders = {}
            for i, local in enumerate(locals):
                dataloaders[i] = DataLoader(DatasetSplit(train_dataset, train_user_groups[i]),
                        batch_size=args.local_bs, shuffle=True, num_workers=4, drop_last=False)
        else:
            dataloaders = {}
            for i, local in enumerate(locals):
                dataloaders[i] = local.train_loader
        calibrate_classifier(
            global_model, None, dataloaders, args.num_classes, args.sample_per_class, args.lr, device)
    elif args.fedexnn_classifer == "multihead":
        pass
    else:
        raise NotImplementedError
    training_pfl_acc_list = []

    # Test global and ensemble model
    # NOTE: global weights need not to be averaged
    logger.info("avg acc:")
    test_acc, test_loss = test(global_model, global_test_loader, device)
    pfl_accs = []

    result_dict = {}
    for idx in range(args.num_users):
        local_test_acc, _ = test(global_model, locals[idx].test_loader, device)
        result_dict["pfl_acc_on_{}".format(users[idx])] = local_test_acc
        pfl_accs.append(local_test_acc)
    if args.backdoor_train:
        ensemble_backdoor_acc, ensemble_backdoor_loss = test(global_model, backdoor_test_loader, device)
        logger.info(f"ensemble_backdoor_acc: {ensemble_backdoor_acc}")
        ExpTool.record({"ensemble_backdoor_acc": ensemble_backdoor_acc,
                        "ensemble_backdoor_loss": ensemble_backdoor_loss})

    logger.info(f"pfl_accs: {pfl_accs}")
    logger.info(f"training_pfl_acc_list:{training_pfl_acc_list}")
    logger.info(f"test_acc:{test_acc}")

    ExpTool.record(result_dict)
    logger.info(f"result_dict: {result_dict}")
    ExpTool.record({"comm_round": args.fedexnn_split_num + 1, "local_epoch": args.local_ep, "train_loss": avg_train_loss, "test_acc": test_acc})
    ExpTool.upload()
    count_para = 0
    for key, value in global_model.named_parameters():
        count_para += value.numel()
    summary_dict = {"count_paras": count_para}
    logger.info(f"global_model's summary_dict: {summary_dict}")
    ExpTool.summary(summary_dict)

    # ===============================================
    if not args.checkpoint == "no":
        ExpTool.save_pickle(global_model.cpu().state_dict(), args.checkpoint, exp_dir=True)
    # torch.save(global_model.cpu().state_dict(), '{}_{}_{}clients_{}.pkl'.format(args.type, args.dataset, args.num_users, args.alpha))
    return global_model

























