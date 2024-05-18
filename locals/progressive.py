from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from helpers.datasets import partition_data
from helpers.utils import get_dataset, average_weights, DatasetSplit, KLDiv, setup_seed, test, progressive_test
from helpers.exp_path import ExpTool

class PnnLocalUpdate(object):
    def __init__(self, args, train_dataset, test_dataset, global_test_loader, train_idxs, test_idxs):
        self.args = args
        self.train_loader = DataLoader(DatasetSplit(train_dataset, train_idxs),
                                       batch_size=self.args.local_bs, shuffle=True, num_workers=4, drop_last=False)
        self.test_loader = DataLoader(DatasetSplit(test_dataset, test_idxs),
                                       batch_size=self.args.local_bs, shuffle=False, num_workers=4, drop_last=False)
        self.global_test_loader = global_test_loader

    def update_weights(self, client_id, epochs, model, device, if_test):
        model.adaptation(str(client_id))
        model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                    momentum=0.9)
        for epoch in tqdm(range(epochs)):
            train_losses = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                # ---------------------------------------
                output = model(images, str(client_id))
                loss = F.cross_entropy(output, labels)
                # ---------------------------------------
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
        if if_test:
            pfl_acc, pfl_test_loss, correct = progressive_test(model, self.test_loader, str(client_id), device)
        else:
            pfl_acc = 0.0
        avg_train_loss = np.mean(train_losses)
        print(f"client_id:{client_id} loss:{loss} ")
        return None, avg_train_loss, None, pfl_acc
