import torch
import torch.nn.functional as F
from torch import nn
import logging
logger = logging.getLogger()

from .pnn_cnn import PNN_ResBase, PNN_ResBasicBlock, PNN_ResBottleneck


class MultiHeadClassifier(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        # self.classifiers = torch.nn.ModuleDict()
        self.classifiers = torch.nn.ModuleList([])

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def adaptation(self, new_in_features, new_out_feautres):
        self.freeze()
        self.classifiers.append(torch.nn.Linear(new_in_features, new_out_feautres))

    def forward(self, xs):
        outputs = []
        for i, classifier in enumerate(self.classifiers):
            outputs.append(classifier(xs[i]))
        return sum(outputs)


class CNNAdapter(nn.Module):
    def __init__(
        self,
        in_features,
        out_features_per_column,
        num_prev_modules,
        activation=F.relu,
    ):
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



class Federated_PNNLayer(nn.Module):

    def __init__(self, in_features, out_features_per_column, is_pool, adapter="cnn"):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column
        self.is_pool = is_pool
        self.adapter = adapter
        self.columns = nn.ModuleList([])

    @property
    def num_columns(self):
        return len(self.columns)

    def adaptation(self):
        self._add_column()


    def _add_column(self):
        """Add a new column."""
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

    def forward(self, x):
        hs = []
        for ii in range(self.num_columns):
            hs.append(self.columns[ii](x[: ii + 1]))
        return hs






class Federated_PNN_CNN(nn.Module):
    def __init__(
        self,
        num_layers=3,
        in_features=3,
        hidden_features_per_column=128,
        num_of_classes=10,
        adapter="cnn",
        classifier_name="progressive",
    ):
        super().__init__()
        assert num_layers >= 1
        assert num_layers == 3
        self.num_layers = num_layers
        self.in_features = in_features
        self.out_features_per_columns = hidden_features_per_column
        self.num_of_classes = num_of_classes

        self.layers = nn.ModuleList()
        self.layers.append(Federated_PNNLayer(in_features, hidden_features_per_column,
            is_pool=True, adapter=adapter))
        for layer_i in range(num_layers - 1):
            if layer_i == num_layers - 2:
                is_pool = False
            else:
                is_pool = True
            lay = Federated_PNNLayer(
                hidden_features_per_column,
                hidden_features_per_column,
                is_pool=is_pool,
                adapter=adapter,
            )
            self.layers.append(lay)

        self.classifier_name = classifier_name
        if self.classifier_name == "fixed":
            self.classifier = torch.nn.Linear(hidden_features_per_column * 4 * 4, num_of_classes)
        elif self.classifier_name == "progressive":
            self.classifier = MultiHeadClassifier()
        else:
            raise NotImplementedError


    def adaptation(self):
        for layer in self.layers:
            layer.adaptation()
        if self.classifier_name == "fixed":
            pass
        elif self.classifier_name == "progressive":
            self.classifier.adaptation(self.out_features_per_columns * 4 * 4, self.num_of_classes)
        else:
            raise RuntimeError


    def forward(self, x):
        # x = x.contiguous()
        # x = x.view(x.size(0), self.in_features)

        num_columns = self.layers[0].num_columns

        x = [x for _ in range(num_columns)]
        for lay in self.layers:
            x = [F.relu(el) for el in lay(x)]

        if self.classifier_name == "fixed":
            x = sum(x)
            x = x.view(x.size(0), -1)
            outputs = self.classifier(x)
        elif self.classifier_name == "progressive":
            x = [xi.view(xi.size(0), -1) for xi in x]
            outputs = self.classifier(x)
        else:
            raise NotImplementedError
        return outputs





class Federated_PNN_ResLayer(nn.Module):

    def __init__(self, in_features, out_features_per_column, block,
                group_norm=0, 
                adapter="cnn"):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column
        self.block = block
        self.group_norm = group_norm
        self.adapter = adapter
        self.columns = nn.ModuleList([])

    @property
    def num_columns(self):
        return len(self.columns)

    def adaptation(self):
        self._add_column()


    def _add_column(self):
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

    def forward(self, x):
        hs = []
        for ii in range(self.num_columns):
            hs.append(self.columns[ii](x[: ii + 1]))
        return hs







class FL_PNN_Resnet(nn.Module):
    def __init__(
        self,
        block, num_blocks, num_classes=10, group_norm=0, res_base_width=64, in_channels=3,
        adapter="cnn",
        classifier_name="progressive",
    ):
        super().__init__()
        self.num_of_classes = num_classes
        self.res_base_width = res_base_width
        self.in_planes = self.res_base_width
        self.group_norm = group_norm
        self.adapter = adapter

        self.num_of_classes = num_classes

        self.expansion = block.expansion

        logger.info(f"num_classes={num_classes}, group_norm={group_norm}, \
                    \n  res_base_width={res_base_width}, in_channels={in_channels}")

        self.layers = nn.ModuleList()
        self.layers.append(Federated_PNN_ResLayer(in_channels, self.res_base_width, PNN_ResBase,
                group_norm=self.group_norm, 
                adapter=self.adapter))
        self._make_layer(block, self.res_base_width, num_blocks[0], stride=1)
        self._make_layer(block, self.res_base_width*2, num_blocks[1], stride=2)
        self._make_layer(block, self.res_base_width*4, num_blocks[2], stride=2)
        self._make_layer(block, self.res_base_width*8, num_blocks[3], stride=2)
        self.classifier_name = classifier_name
        if self.classifier_name == "fixed":
            self.classifier = torch.nn.Linear(res_base_width * 8 * self.expansion, self.num_of_classes)
        elif self.classifier_name == "progressive":
            self.classifier = MultiHeadClassifier()
        else:
            raise NotImplementedError


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            self.layers.append(Federated_PNN_ResLayer(self.in_planes, planes, block,
                group_norm=self.group_norm, 
                adapter=self.adapter))
            self.in_planes = planes * block.expansion


    def adaptation(self):
        for layer in self.layers:
            layer.adaptation()
        if self.classifier_name == "fixed":
            pass
        elif self.classifier_name == "progressive":
            self.classifier.adaptation(self.res_base_width * 8 * self.expansion, self.num_of_classes)
        else:
            raise NotImplementedError


    def forward(self, x):
        num_columns = self.layers[0].num_columns
        x = [x for _ in range(num_columns)]
        for lay in self.layers:
            x = [F.relu(el) for el in lay(x)]
        # logger.info(f"x.shapes:{[xi.shape for xi in x]}")
        if self.classifier_name == "fixed":
            x = sum(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            outputs = self.classifier(x)
        elif self.classifier_name == "progressive":
            x = [F.adaptive_avg_pool2d(xi, (1, 1)) for xi in x]
            x = [xi.view(xi.size(0), -1) for xi in x]
            outputs = self.classifier(x)
        else:
            raise NotImplementedError
        return outputs




def fl_pnn_resnet18(num_classes=10, **kwargs):
    return FL_PNN_Resnet(PNN_ResBasicBlock, [2, 2, 2, 2], num_classes, **kwargs)


def fl_pnn_resnet34(num_classes=10, **kwargs):
    return FL_PNN_Resnet(PNN_ResBasicBlock, [3, 4, 6, 3], num_classes, **kwargs)


def fl_pnn_resnet50(num_classes=10, **kwargs):
    return FL_PNN_Resnet(PNN_ResBottleneck, [3, 4, 6, 3], num_classes, **kwargs)


def fl_pnn_resnet101(num_classes=10, **kwargs):
    return FL_PNN_Resnet(PNN_ResBottleneck, [3, 4, 23, 3], num_classes, **kwargs)


def fl_pnn_resnet152(num_classes=10, **kwargs):
    return FL_PNN_Resnet(PNN_ResBottleneck, [3, 8, 36, 3], num_classes, **kwargs)




















