from tqdm import tqdm
import numpy as np
from itertools import chain
import logging

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from helpers.datasets import partition_data
from helpers.utils import get_dataset, average_weights, DatasetSplit, BackdoorDS, KLDiv, setup_seed, test, progressive_test
from helpers.exp_path import ExpTool
from locals.cl_loss.info_nce import INFONCE


class LocalUpdate(object):
    def __init__(self, args, train_dataset, test_dataset, global_test_loader, train_idxs, test_idxs, init_model,
                 backdoor_train=False):
        self.args = args
        self.model = init_model
        self.backdoor_train = backdoor_train
        if backdoor_train:
            self.train_loader = DataLoader(BackdoorDS(DatasetSplit(train_dataset, train_idxs), args.backdoor_size),
                                        batch_size=self.args.local_bs, shuffle=True, num_workers=4, drop_last=False)
        else:
            self.train_loader = DataLoader(DatasetSplit(train_dataset, train_idxs),
                                        batch_size=self.args.local_bs, shuffle=True, num_workers=4, drop_last=False)
        self.test_loader = DataLoader(DatasetSplit(test_dataset, test_idxs),
                                       batch_size=self.args.local_bs, shuffle=False, num_workers=4, drop_last=False)
        self.global_test_loader = global_test_loader
        self.info_nce = INFONCE(args.contrastive_n_views)

    def load_global_model(self):
        pass

    def add_CL_head(self, CL_head):
        self.CL_head = CL_head




    def update_weights(self, client_id, epochs, device, if_test):
        model = self.model
        model.train()
        model.to(device)
        correct = 0
        if self.args.contrastive_train:
            self.info_nce.to(device)
            self.CL_head.to(device)
            optimizer = torch.optim.SGD(chain(model.parameters(), self.CL_head.parameters()), lr=self.args.lr,
                                        momentum=0.9)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=0.9)
        # label_list = [0] * 100
        # for batch_idx, (images, labels) in enumerate(self.train_loader):
        #     for i in range(100):
        #         label_list[i] += torch.sum(labels == i).item()
        # print(label_list)
        for epoch in tqdm(range(epochs)):
            train_losses = []
            correct = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                model.zero_grad()
                if self.args.contrastive_train:
                    labels = torch.cat([labels for _ in range(self.args.contrastive_n_views)], dim=0).to(device)
                    # images = [image.to(device) for image in images]
                    # ---------------------------------------
                    images = torch.cat(images, dim=0).to(device)
                    outputs, logits = model(images, get_logits=True)
                    z = self.CL_head(logits)
                    cls_loss = F.cross_entropy(outputs, labels)
                    posi_nega_logits, posi_nega_labels = self.info_nce(z)
                    CL_loss = F.cross_entropy(posi_nega_logits, posi_nega_labels)
                    loss = cls_loss + CL_loss * self.args.contrastive_weight

                    # ---------------------------------------
                else:
                    images, labels = images.to(device), labels.to(device)
                    # ---------------------------------------
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)
                    # ---------------------------------------

                pred = torch.max(outputs, 1)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()    

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_pfl_acc = 100. * correct / len(self.train_loader.dataset)
            avg_train_loss = np.mean(train_losses)
            logging.info(f"client_id:{client_id} epoch:[{epoch}/{epochs}] loss:{loss}, train_loss:{avg_train_loss}, train_pfl_acc: {train_pfl_acc}")
        if if_test:
            acc, test_loss = test(model, self.global_test_loader, device)
            pfl_acc, pfl_test_loss = test(model, self.test_loader, device)
        else:
            acc = 0.0
            pfl_acc = 0.0
        model.to("cpu")
        return model.state_dict(), avg_train_loss, acc, pfl_acc, train_pfl_acc










