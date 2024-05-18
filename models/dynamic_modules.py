import torch
from torch.nn import Module
from typing import Optional


class MultiTaskModule(Module):
    def __init__(self):
        super().__init__()
        self.known_train_tasks = set()


    def adaptation(self, task_id):
        if self.training:
            self.train_adaptation(task_id)
        else:
            self.eval_adaptation(task_id)

    def eval_adaptation(self, task_id):
        pass

    def train_adaptation(self, task_id):
        # """Update known task labels."""
        self.known_train_tasks.add(task_id)

    def forward(self, x, task_id=None) -> torch.Tensor:
        if task_id is None:
            return self.forward_all_tasks(x)
        else:
            return self.forward_single_task(x, task_id)

    def forward_single_task(self, x, task_id=None) -> torch.Tensor:
        raise NotImplementedError()

    def forward_all_tasks(self, x):
        res = {}
        for task_id in self.known_train_tasks:
            res[task_id] = self.forward_single_task(x, task_id)
        return res



class MultiHeadClassifier(MultiTaskModule):
    def __init__(
        self,
    ):
        """Init.

        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        :param masking: whether unused units should be masked (default=True).
        :param mask_value: the value used for masked units (default=-1000).
        """
        super().__init__()
        self.classifiers = torch.nn.ModuleDict()

    def adaptation(self, task_id, new_in_features, new_out_feautres):
        super().adaptation(task_id)
        self.classifiers[task_id] = torch.nn.Linear(new_in_features, new_out_feautres)

    def forward_single_task(self, x, task_id):
        task_id = str(task_id)
        out = self.classifiers[task_id](x)
        return out

    def forward_all_tasks(self, x: torch.Tensor):
        # res = {}
        # for task_id in self.known_train_tasks:
        #     res[task_id] = self.forward_single_task(x, task_id)
        outputs = []
        for task_id in self.known_train_tasks:
            outputs.append(self.forward_single_task(x, task_id))
        return sum(outputs)



