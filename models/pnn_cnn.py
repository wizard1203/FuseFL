import torch
import torch.nn.functional as F
from torch import nn
import logging
logger = logging.getLogger()

from .dynamic_modules import MultiTaskModule, MultiHeadClassifier

from .resnet import norm2d

# from .resnet import BasicBlock, Bottleneck


class CNNAdapter(nn.Module):
    def __init__(
        self,
        in_features,
        out_features_per_column,
        num_prev_modules,
        activation=F.relu,
    ):
        """
        :param in_features: size of each input sample
        :param out_features_per_column: size of each output sample
        :param num_prev_modules: number of previous modules
        :param activation: activation function (default=ReLU)
        """
        super().__init__()
        self.num_prev_modules = num_prev_modules
        self.activation = activation

        if num_prev_modules == 0:
            return  # first adapter is empty

        # Eq. 2 - CNN adapter. Not needed for the first task.
        self.V = nn.Conv2d(in_features * num_prev_modules, out_features_per_column, 1)
        self.alphas = nn.Parameter(torch.randn(num_prev_modules))
        self.U = nn.Conv2d(out_features_per_column, out_features_per_column, 1)

    def forward(self, x):
        if self.num_prev_modules == 0:
            return 0  # first adapter is empty

        assert len(x) == self.num_prev_modules
        # assert len(x[0].shape) == 2, (
        #     "Inputs to CNNAdapter should have two dimensions: "
        #     "<batch_size, channels, height, width>."
        # )
        for i, el in enumerate(x):
            x[i] = self.alphas[i] * el
            # logger.info(f"i:{i} x[i].shapes:{x[i].shape}")
        x = torch.cat(x, dim=1)
        x = self.U(self.activation(self.V(x)))
        # logger.info(f"x.shapes:{x.shape}")
        return x



class PNN_CNNColumn(nn.Module):
    def __init__(
        self,
        in_features,
        out_features_per_column,
        num_prev_modules,
        is_pool,
        adapter="cnn",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column
        self.num_prev_modules = num_prev_modules
        self.is_pool = is_pool

        self.adapter = CNNAdapter(
            in_features, in_features, num_prev_modules
        )
        if self.is_pool:
            self.itoh = nn.Sequential(
                nn.Conv2d(in_features, out_features_per_column, 3),
                nn.BatchNorm2d(out_features_per_column),
                nn.ReLU(),
                nn.MaxPool2d(2, 2))
        else:
            self.itoh = nn.Sequential(
                nn.Conv2d(in_features, out_features_per_column, 3),
                nn.BatchNorm2d(out_features_per_column),
                nn.ReLU())

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        prev_xs, last_x = x[:-1], x[-1]
        hs = self.adapter(prev_xs)
        # if isinstance(hs, int):
        #     pass
        # else:
        #     logger.info(f"hs.shapes:{hs.shape}")
        # logger.info(f"last_x.shape:{last_x.shape}")
        h_mix = hs + last_x
        h = self.itoh(h_mix)
        return h


class PNN_CNNLayer(MultiTaskModule):
    """Progressive Neural Network layer.
    """

    def __init__(self, in_features, out_features_per_column, is_pool, adapter="cnn"):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column
        self.is_pool = is_pool
        self.adapter = adapter

        # convert from task label to module list order
        self.task_to_module_idx = {}
        self.columns = nn.ModuleList([])

    @property
    def num_columns(self):
        return len(self.columns)

    def adaptation(self, task_id):
        super().train_adaptation(task_id)
        self.task_to_module_idx[task_id] = self.num_columns
        self._add_column(task_id)


    def _add_column(self, task_id):
        for param in self.parameters():
            param.requires_grad = False
        self.columns.append(
            PNN_CNNColumn(
                self.in_features,
                self.out_features_per_column,
                self.num_columns,
                self.is_pool,
                adapter=self.adapter,
            )
        )

    def forward_single_task(self, x, task_id):
        col_idx = self.task_to_module_idx[task_id]
        hs = []
        for ii in range(col_idx + 1):
            hs.append(self.columns[ii](x[: ii + 1]))
        return hs


class PNN_CNN(MultiTaskModule):
    def __init__(
        self,
        num_layers=3,
        in_features=3,
        hidden_features_per_column=128,
        num_of_classes=10,
        adapter="cnn",
    ):
        super().__init__()
        assert num_layers >= 1
        assert num_layers == 3
        self.num_layers = num_layers
        self.in_features = in_features
        self.out_features_per_columns = hidden_features_per_column
        self.num_of_classes = num_of_classes

        self.layers = nn.ModuleList()
        self.layers.append(PNN_CNNLayer(
            in_features, hidden_features_per_column,
            is_pool=True, adapter=adapter))
        for layer_i in range(num_layers - 1):
            if layer_i == num_layers - 2:
                is_pool = False
            else:
                is_pool = True
            lay = PNN_CNNLayer(
                hidden_features_per_column,
                hidden_features_per_column,
                is_pool=is_pool,
                adapter=adapter,
            )
            self.layers.append(lay)
        self.classifier = MultiHeadClassifier()


    def adaptation(self, task_id):
        super().adaptation(task_id)
        for layer in self.layers:
            layer.adaptation(task_id)
        self.classifier.adaptation(task_id, self.out_features_per_columns * 4 * 4, self.num_of_classes)


    def forward_single_task(self, x, task_id):
        # x = x.contiguous()
        # x = x.view(x.size(0), self.in_features)

        num_columns = self.layers[0].num_columns
        col_idx = self.layers[-1].task_to_module_idx[task_id]

        x = [x for _ in range(num_columns)]
        for lay in self.layers:
            x = [F.relu(el) for el in lay(x, task_id)]
        # logger.info(f"x.shapes:{[xi.shape for xi in x]}")
        col_x = x[col_idx]
        col_x = col_x.view(col_x.size(0), -1)
        return self.classifier(col_x, task_id)



class PNN_ResBase(nn.Module):
    # def __init__(self, in_channels=3, res_base_width=64, group_norm=0,
    #              num_prev_modules=1, adapter="cnn"):
    def __init__(self, in_planes, planes, stride=1, group_norm=0,
                 num_prev_modules=1, adapter="cnn"):
        super(PNN_ResBase, self).__init__()
        # self.in_planes = 64
        # self.res_base_width = res_base_width
        # self.in_planes = self.res_base_width

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm2d(planes, group_norm)
        self.adapter = CNNAdapter(
            in_planes, in_planes, num_prev_modules
        )


    def forward(self, x):
        prev_xs, last_x = x[:-1], x[-1]
        hs = self.adapter(prev_xs)
        h_mix = hs + last_x

        out = self.conv1(h_mix)
        out = self.bn1(out)
        out = F.relu(out)

        return out


    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False




class PNN_ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=0,
                num_prev_modules=1, adapter="cnn"):
        super(PNN_ResBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm2d(planes, group_norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm2d(planes, group_norm)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.num_prev_modules = num_prev_modules
        self.adapter = CNNAdapter(
            in_planes, in_planes, num_prev_modules
        )


    def forward(self, x):
        prev_xs, last_x = x[:-1], x[-1]
        hs = self.adapter(prev_xs)
        h_mix = hs + last_x

        out = F.relu(self.bn1(self.conv1(h_mix)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(h_mix)
        out = F.relu(out)

        return out

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False



class PNN_ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, group_norm=0,
                num_prev_modules=1, adapter="cnn"):
        super(PNN_ResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm2d(planes, group_norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm2d(planes, group_norm)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = norm2d(planes * self.expansion, group_norm)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.adapter = CNNAdapter(
            in_planes, in_planes, num_prev_modules
        )

    def forward(self, x):
        prev_xs, last_x = x[:-1], x[-1]
        hs = self.adapter(prev_xs)
        h_mix = hs + last_x

        out = F.relu(self.bn1(self.conv1(h_mix)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(h_mix)
        out = F.relu(out)
        return out


class PNN_ResLayer(MultiTaskModule):
    def __init__(self, in_features, out_features_per_column, block,
                group_norm=0, 
                adapter="cnn"):
        super().__init__()
        self.block = block
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column
        self.group_norm = group_norm
        self.adapter = adapter
        self.task_to_module_idx = {}
        self.columns = nn.ModuleList([])

    @property
    def num_columns(self):
        return len(self.columns)

    def adaptation(self, task_id):
        super().train_adaptation(task_id)
        self.task_to_module_idx[task_id] = self.num_columns
        self._add_column(task_id)


    def _add_column(self, task_id):
        """Add a new column."""
        for param in self.parameters():
            param.requires_grad = False
        self.columns.append(
            self.block(
                in_planes=self.in_features, planes=self.out_features_per_column,
                stride=1, group_norm=self.group_norm,
                num_prev_modules=self.num_columns,
                adapter=self.adapter,
            )
        )

    def forward_single_task(self, x, task_id):
        col_idx = self.task_to_module_idx[task_id]
        hs = []
        for ii in range(col_idx + 1):
            hs.append(self.columns[ii](x[: ii + 1]))
        return hs




class PNN_Resnet(MultiTaskModule):
    def __init__(
        self,
        block, num_blocks, num_classes=10, group_norm=0, res_base_width=64, in_channels=3,
        adapter="cnn",
    ):
        super().__init__()
        self.num_of_classes = num_classes
        self.res_base_width = res_base_width
        self.in_planes = self.res_base_width
        self.group_norm = group_norm
        self.adapter = adapter

        self.expansion = block.expansion

        logger.info(f"num_classes={num_classes}, group_norm={group_norm}, \
                    \n  res_base_width={res_base_width}, in_channels={in_channels}")

        # PNN_ResBase(in_channels=in_channels, res_base_width=res_base_width, group_norm=group_norm,
        #         num_prev_modules=num_prev_modules, adapter=adapter)
        self.layers = nn.ModuleList()
        self.layers.append(PNN_ResLayer(in_channels, self.res_base_width, PNN_ResBase,
                group_norm=self.group_norm, 
                adapter=self.adapter))
        self._make_layer(block, self.res_base_width, num_blocks[0], stride=1)
        self._make_layer(block, self.res_base_width*2, num_blocks[1], stride=2)
        self._make_layer(block, self.res_base_width*4, num_blocks[2], stride=2)
        self._make_layer(block, self.res_base_width*8, num_blocks[3], stride=2)
        self.classifier = MultiHeadClassifier()


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        # layers = []
        # for stride in strides:
        #     layers.append(block(self.in_planes, planes, stride))
        #     self.in_planes = planes * block.expansion
        # return nn.Sequential(*layers)
        for stride in strides:
            self.layers.append(PNN_ResLayer(self.in_planes, planes, block,
                group_norm=self.group_norm, 
                adapter=self.adapter))
            self.in_planes = planes * block.expansion


    def adaptation(self, task_id):
        super().adaptation(task_id)
        for layer in self.layers:
            layer.adaptation(task_id)
        self.classifier.adaptation(task_id, self.res_base_width * 8 * self.expansion, self.num_of_classes)


    def forward_single_task(self, x, task_id):
        # x = x.contiguous()
        # x = x.view(x.size(0), self.in_features)

        num_columns = self.layers[0].num_columns
        col_idx = self.layers[-1].task_to_module_idx[task_id]

        x = [x for _ in range(num_columns)]
        for lay in self.layers:
            x = [F.relu(el) for el in lay(x, task_id)]
        # logger.info(f"x.shapes:{[xi.shape for xi in x]}")
        col_x = x[col_idx]

        col_x = F.adaptive_avg_pool2d(col_x, (1, 1))
        col_x = col_x.view(col_x.size(0), -1)
        return self.classifier(col_x, task_id)





def pnn_resnet18(num_classes=10, **kwargs):
    return PNN_Resnet(PNN_ResBasicBlock, [2, 2, 2, 2], num_classes, **kwargs)


def pnn_resnet34(num_classes=10, **kwargs):
    return PNN_Resnet(PNN_ResBasicBlock, [3, 4, 6, 3], num_classes, **kwargs)


def pnn_resnet50(num_classes=10, **kwargs):
    return PNN_Resnet(PNN_ResBottleneck, [3, 4, 6, 3], num_classes, **kwargs)


def pnn_resnet101(num_classes=10, **kwargs):
    return PNN_Resnet(PNN_ResBottleneck, [3, 4, 23, 3], num_classes, **kwargs)


def pnn_resnet152(num_classes=10, **kwargs):
    return PNN_Resnet(PNN_ResBottleneck, [3, 8, 36, 3], num_classes, **kwargs)








