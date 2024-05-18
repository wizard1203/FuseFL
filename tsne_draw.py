import argparse
import logging
import os
import random
import socket
import sys
import yaml
import copy

import torch
import numpy as np

from torchvision import transforms
from torchvision.utils import save_image
import torch.utils.data as data

import matplotlib
import matplotlib.pyplot as plt

from dim_reducer import Dim_Reducer


color_map = [
"#252525",
"#e41a1c",
"#337eb8",
"#4daf4a",
"#542788",
"#ff7f00",
"#ffff33",
"#a65628",
"#f781bf",
"#1711ff",
]

# noise_color = "#737373"
noise_color = "#252525"


size_map = [
"o",
"^",
"*",
"x",
"+",
]


def draw_scatter(X_list, Y_list, classes, file_name):

    fig = plt.figure(figsize=(8, 8))
    # fig = plt.figure(figsize=(64, 64))
    # fig = plt.figure(figsize=(32, 32))
    size_scale = 1
    ax = fig.gca()

    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticks([])
    ax.set_yticks([])

    # real_size = 40
    # alpha = 0.8

    real_size = 80 * size_scale
    alpha = 0.5

    # plt.subplots_adjust(**dict(bottom=0.08, left=0.1, right=0.96, top=0.8))
    plt.subplots_adjust(**dict(bottom=0.01, left=0.01, right=0.99, top=0.99))

    X = X_list
    Y = Y_list

    for i in range(classes):
        label = f"Label {i}"
        # edgecolors = matplotlib.colors.colorConverter.to_rgba(color_map[i], alpha=alpha)
        # plt.scatter(X[:, 0][Y == i], X[:, 1][Y == i], s=real_size, marker=size_map[i], c=color_map[i], edgecolors=edgecolors,
        #             label=label, alpha=alpha)
        plt.scatter(X[:, 0][Y == i], X[:, 1][Y == i], s=real_size, c=color_map[i], 
                    label=label, alpha=alpha)
    plt.legend(frameon=False, ncol=2)
    # ax.set_rasterized(True)
    # plt.savefig(file_name, transparent=True, bbox_inches='tight')
    plt.savefig(file_name, transparent=True)



def draw_tsne(device, max_classes, features, labels, tSNE_save_path):
    dim_reducer = Dim_Reducer(device=device)

    data_tsne, labels = dim_reducer.unsupervised_reduce(reduce_method="tSNE", 
        model=None, batch_data=(features, labels), data_loader=None, num_points=features.shape[0])
    # logging.info(f"Epoch: {epoch}, client_index: {client_index}, data_tsne.shape:{data_tsne.shape} ")

    # split_data_tsne = torch.split(data_tsne, num_points_per_clients, dim=0)
    # split_label = torch.split(labels, num_points_per_clients, dim=0)
    draw_scatter(data_tsne, labels, max_classes, tSNE_save_path)


    # plt.figure(figsize=(6, 4))
    # fig, ax = plt.subplot(1, 2, 1)
    # fig.set_figheight(5)
    # fig.set_figwidth(7)

    # plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, alpha=0.6, 
    #             cmap=plt.cm.get_cmap('rainbow', 10))
    # plt.title("t-SNE")
    # plt.savefig(tSNE_save_path)


























