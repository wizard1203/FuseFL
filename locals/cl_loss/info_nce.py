import logging

import torch
import torch.nn.functional as F

class INFONCE(torch.nn.Module):
    def __init__(self, n_views, temperature=1.0):
        super(INFONCE, self).__init__()
        self.n_views = n_views
        # self.device = device
        self.temperature = temperature

    def to(self, device):
        self.device = device


    def forward(self, features):
        # labels = torch.cat([torch.arange(self.batch_size) for _ in range(self.n_views)], dim=0)
        labels = torch.cat([torch.arange(int(features.shape[0] / self.n_views)) for _ in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        # logging.info(f"features.shape:{features.shape}, labels:{labels.shape}")

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels




