import logging

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def compute_classes_mean_cov(global_model, models, dataloaders, num_classes, device):
    features_means, features_covs, features_count = [], [], []
    for client_idx in dataloaders.keys():
        if global_model is None:
            means, covs, sizes = get_means_covs_from_client(
                models[client_idx], dataloaders[client_idx], device, num_classes
            )
        else:
            means, covs, sizes = get_means_covs_from_client(
                global_model, dataloaders[client_idx], device, num_classes
            )

        features_means.append(means)
        features_covs.append(covs)
        features_count.append(sizes)

    num_classes = len(features_count[0])
    labels_count = [sum(cnts) for cnts in zip(*features_count)]
    classes_mean = []
    for c, (means, sizes) in enumerate(
        zip(zip(*features_means), zip(*features_count))
    ):
        weights = torch.tensor(sizes, device=device) / labels_count[c]
        means_ = torch.stack(means, dim=-1)
        classes_mean.append(torch.sum(means_ * weights, dim=-1))
    classes_cov = [None for _ in range(num_classes)]
    for c in range(num_classes):
        # for k in self.train_clients:
        for client_idx in dataloaders.keys():
            if classes_cov[c] is None:
                classes_cov[c] = torch.zeros_like(features_covs[client_idx][c])

            classes_cov[c] += (features_count[client_idx][c] - 1) / (
                labels_count[c] - 1
            ) * features_covs[client_idx][c] + (
                features_count[client_idx][c] / (labels_count[c] - 1)
            ) * (
                features_means[client_idx][c].unsqueeze(1)
                @ features_means[client_idx][c].unsqueeze(0)
            )

        classes_cov[c] -= (labels_count[c] / labels_count[c] - 1) * (
            classes_mean[c].unsqueeze(1) @ classes_mean[c].unsqueeze(0)
        )

    return classes_mean, classes_cov

def generate_virtual_representation(
    classes_mean, classes_cov, sample_per_class, device
):
    data, targets = [], []
    for c, (mean, cov) in enumerate(zip(classes_mean, classes_cov)):
        samples = np.random.multivariate_normal(
            mean.cpu().numpy(), cov.cpu().numpy(), sample_per_class
        )
        data.append(torch.tensor(samples, dtype=torch.float, device=device))
        targets.append(
            torch.ones(
                sample_per_class, dtype=torch.long, device=device
            )
            * c
        )

    data = torch.cat(data)
    targets = torch.cat(targets)
    return data, targets

def calibrate_classifier(global_model, models, dataloaders, num_classes, sample_per_class, local_lr, device):
    classes_mean, classes_cov = compute_classes_mean_cov(global_model, models, dataloaders, num_classes, device)
    data, targets = generate_virtual_representation(classes_mean, classes_cov, sample_per_class, device)

    class RepresentationDataset(Dataset):
        def __init__(self, data, targets):
            self.data = data
            self.targets = targets

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

        def __len__(self):
            return len(self.targets)

    dataset = RepresentationDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(global_model.parameters(), lr=local_lr)

    for x, y in dataloader:
        logits = global_model.classifier(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




def get_means_covs_from_client(
    model, dataloader, device, num_classes
):
    features = []
    targets = []
    feature_length = None
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        features.append(model.get_final_features(x))
        targets.append(y)
        # features.append(model.get_final_features(x).to("cpu"))
        # targets.append(y.to("cpu"))
        # logging.info(f"features shape: {features[-1].shape}")

    targets = torch.cat(targets)
    features = torch.cat(features)
    feature_length = features.shape[-1]
    # indices = [
    #     torch.where(targets == i)[0]
    #     for i in range(len(num_classes))
    # ]
    indices = [
        torch.where(targets == i)[0]
        for i in range(num_classes)
    ]
    classes_features = [features[idxs] for idxs in indices]
    classes_means, classes_covs = [], []
    for fea in classes_features:
        if fea.shape[0] > 0:
            classes_means.append(fea.mean(dim=0))
            # classes_covs.append(fea.t().cov(correction=0))
            classes_covs.append(torch.cov(fea.t(), correction=0, fweights=None, aweights=None))
        else:
            classes_means.append(torch.zeros(feature_length, device=device))
            classes_covs.append(
                torch.zeros(feature_length, feature_length, device=device)
            )
    return classes_means, classes_covs, [len(idxs) for idxs in indices]










