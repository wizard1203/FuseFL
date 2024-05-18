import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import random

from .cl_dataset import _get_simclr_pipeline_transform, _get_cl_transform


def get_image_size(dataset):
    image_size = {
        "mnist": 28,
        "fmnist": 28,
        "SVHN": 32,
        "cifar10": 32,
        "cifar100": 32,
        "Tiny-ImageNet-200": 64,
    }[dataset]
    return image_size

def get_num_of_labels(dataset):
    num_of_labels = {
        "mnist": 10,
        "fmnist": 10,
        "SVHN": 10,
        "cifar10": 10,
        "cifar100": 100,
        "Tiny-ImageNet-200": 200,
    }[dataset]
    return num_of_labels


def load_data(image_size, dataset, datadir, contrastive_train=False, contrastive_n_views=2, **kwargs):
    # data_dir = '/dataset'
    data_dir = datadir
    if contrastive_train:
        contrastive_transform = _get_cl_transform(size=image_size, n_views=contrastive_n_views)

    if dataset == "mnist":
        if contrastive_train:
            train_dataset = datasets.MNIST(data_dir, train=True,
                                        transform=contrastive_transform)
        else:
            train_dataset = datasets.MNIST(data_dir, train=True,
                                        transform=transforms.Compose(
                                            [transforms.ToTensor()]))
        test_dataset = datasets.MNIST(data_dir, train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]))
    elif dataset == "fmnist":
        if contrastive_train:
            train_dataset = datasets.FashionMNIST(data_dir, train=True,
                                                transform=contrastive_transform)
        else:
            train_dataset = datasets.FashionMNIST(data_dir, train=True,
                                                transform=transforms.Compose(
                                                    [transforms.ToTensor()]))
        test_dataset = datasets.FashionMNIST(data_dir, train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                             ]))
    elif dataset == "SVHN":
        if contrastive_train:
            train_dataset = datasets.SVHN(data_dir, split="train",
                                        transform=contrastive_transform)
        else:
            train_dataset = datasets.SVHN(data_dir, split="train",
                                        transform=transforms.Compose(
                                            [transforms.ToTensor()]))
        test_dataset = datasets.SVHN(data_dir, split="test",
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ]))
    elif dataset == "cifar10":
        if contrastive_train:
            train_dataset = datasets.CIFAR10(data_dir, train=True,download=True,
                                            transform=contrastive_transform)
        else:
            train_dataset = datasets.CIFAR10(data_dir, train=True,download=True,
                                            transform=transforms.Compose(
                                                [
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                ]))
        test_dataset = datasets.CIFAR10(data_dir, train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ]))
    elif dataset == "cifar100":
        if contrastive_train:
            train_dataset = datasets.CIFAR100(data_dir, train=True,
                                            transform=contrastive_transform)
        else:
            train_dataset = datasets.CIFAR100(data_dir, train=True,
                                            transform=transforms.Compose(
                                                [
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                ]))
        test_dataset = datasets.CIFAR100(data_dir, train=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                         ]))

    elif dataset == "Tiny-ImageNet-200":
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ])
        }
        if contrastive_train:
            data_transforms["train"] = contrastive_transform
        data_dir = "data/tiny-imagenet-200/"
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val', 'test']}
        train_dataset = image_datasets['train']
        test_dataset = image_datasets['val']
    else:
        raise NotImplementedError
    if dataset == "SVHN":
        X_train, y_train = train_dataset.data, train_dataset.labels
        X_test, y_test = test_dataset.data, test_dataset.labels
    else:
        X_train, y_train = train_dataset.data, train_dataset.targets
        X_test, y_test = test_dataset.data, test_dataset.targets
    if "cifar10" in dataset or dataset == "SVHN":
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    else:
        X_train = X_train.data.numpy()
        y_train = y_train.data.numpy()
        X_test = X_test.data.numpy()
        y_test = y_test.data.numpy()

    return X_train, y_train, X_test, y_test, train_dataset, test_dataset




def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    global_unq = np.unique(y_train, return_counts=False)
    # print(f"global_unq: {global_unq}")
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        # print(f"unq: {unq}, unq_cnt:{unq_cnt}")
        tmp = {}
        for label in global_unq:
            if label not in unq:
                tmp[label] = np.array(0)
            else:
                # tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
                label_idx = np.where(unq == label)
                tmp[label] = unq_cnt[label_idx]
        net_cls_counts[net_i] = tmp


    # print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(image_size, dataset, datadir, partition, alpha=0.4, num_users=5, **kwargs):
    n_parties = num_users
    X_train, y_train, X_test, y_test, train_dataset, test_dataset = load_data(
        image_size, dataset, datadir, **kwargs)
    data_size = y_train.shape[0]

    if partition == "iid":
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "dirichlet":
        min_size = 0
        min_require_size = 10
        label = np.unique(y_test).shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(label):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)  # shuffle the label
                # random [0.5963643 , 0.03712018, 0.04907753, 0.1115522 , 0.2058858 ]
                proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
                proportions = np.array(   # 0 or x
                    [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    test_net_dataidx_map = generate_personalized_data(X_test, y_test, train_data_cls_counts)
    test_data_cls_counts = record_net_data_stats(y_test, test_net_dataidx_map)

    np_test_cls_counts = np.array([list(item.values()) for item in test_data_cls_counts.values()])
    # print("test_data_cls_counts: \n", test_data_cls_counts)
    # print("np_test_cls_counts: \n", np_test_cls_counts)

    return train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts, test_net_dataidx_map, test_data_cls_counts


def generate_personalized_data(X_test, y_test, train_data_cls_counts):
    # train_label = [i.dataset.targets for i in self.train_data_local_dict.values()]
    # label = np.unique(y_test).shape[0]

    # print(train_label[0].shape)
    # class_propotion=np.array([[np.sum(y==i) for i in range(num_classes)] for y in train_label])
    # print(class_propotion)
    # num_train=np.sum(class_propotion)
    # num_class=np.sum(class_propotion, axis=0, keepdims=False)

    num_classes = len(np.unique(y_test))
    n_parties = len(train_data_cls_counts)
    np_train_cls_counts = np.array([list(item.values()) for item in train_data_cls_counts.values()])
    num_class=np.sum(np_train_cls_counts, axis=0, keepdims=False)

    # num_class = [0] * n_parties
    # for i_class in range(num_classes):
    #     for idx in train_data_cls_counts.keys():
    #         num_class[i_class] += train_data_cls_counts[idx][i_class]

    # new_loader=list(zip(X, y))
    # num_test=len(y_test)
    # min_size=0

    # print("train_data_cls_counts: \n", train_data_cls_counts)
    # print("np_train_cls_counts: \n", np_train_cls_counts)
    idx_batch = [[] for _ in range(n_parties)]
    print("num_classes: \n", num_class)

    for k in range(num_classes):
        idx_k = np.where(y_test == k)[0]
        num=len(idx_k)
        np.random.shuffle(idx_k)
        k_num=(np.cumsum(np_train_cls_counts[:, k]*1.0/num_class[k])*num).astype(int)[:-1]
        # print("k_num:", k_num)
        # print("spilt result::::",np.split(idx_k, k_num))
        idx_batch=[idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, k_num))]
    #     print("len(idx_batch)", len(idx_batch))
    # print("[len(idx_j) for idx_j in idx_batch]", [len(idx_j) for idx_j in idx_batch])
    # print("sum of len(idx_j)", sum([len(idx_j) for idx_j in idx_batch]))
    net_dataidx_map = {}

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    # np.save(f"result/{self.args.dataset}_{self.args.partition_alpha}alpha_{self.args.client_num_in_total}client_testdata_cls_matrix", testdata_cls_matrix)
    return net_dataidx_map









