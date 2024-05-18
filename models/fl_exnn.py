from copy import deepcopy
import logging
logger = logging.getLogger()

import torch
import torch.nn.functional as F
from torch import nn

from .resnet import norm2d, BasicBlock, Bottleneck


class MLP_Block(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, actv="relu", num_layers=2):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        layer = nn.Linear(in_features, hidden_features)
        self.layers.append(layer)
        for _ in range(self.num_layers - 1):
            layer = nn.Linear(hidden_features, hidden_features)
            self.layers.append(layer)


    def forward(self, x):
        for lay in self.layers:
            x = F.relu(lay(x))
        return x


class CNN_Block(nn.Module):
    def __init__(self, in_features, hidden_features, is_pool, actv="relu"):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        # self.out_features = out_features
        self.is_pool = is_pool
        self.layers = nn.ModuleList()
        # self.num_layers = num_layers
        # layer = nn.Linear(in_features, hidden_features)
        # self.layers.append(layer)
        # for _ in range(self.num_layers - 1):
        #     layer = nn.Linear(hidden_features, hidden_features)
        #     self.layers.append(layer)
        if self.is_pool:
            self.layer = nn.Sequential(
                nn.Conv2d(in_features, hidden_features, 3),
                nn.BatchNorm2d(hidden_features),
                nn.ReLU(),
                nn.MaxPool2d(2, 2))
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_features, hidden_features, 3),
                nn.BatchNorm2d(hidden_features),
                nn.ReLU())

    def forward(self, x):
        x = self.layer(x)
        return x



def define_fl_exnn_res_layers(
    fedexnn_split_num, block, num_blocks, group_norm=0, res_base_width=64, in_channels=3,
):
    in_planes = res_base_width
    group_norm = group_norm
    # in_planes = in_planes * block.expansion

    def _make_layer(block, planes, num_blocks, stride, group_norm):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        nonlocal in_planes
        for stride in strides:
            layers.append(block(in_planes=in_planes, planes=planes,
                stride=stride, group_norm=group_norm,
            ))
            in_planes = planes * block.expansion

        return layers

    if fedexnn_split_num == 2:
        all_layers = []
        local_layer = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels, res_base_width, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(res_base_width, group_norm),
                nn.ReLU(),
            ),
            nn.Sequential(*_make_layer(block, res_base_width, num_blocks[0], stride=1, group_norm=group_norm)),
            nn.Sequential(*_make_layer(block, res_base_width*2, num_blocks[1], stride=2, group_norm=group_norm)),
        )
        all_layers.append(local_layer)

        local_layer = nn.Sequential(
            nn.Sequential(*_make_layer(block, res_base_width*4, num_blocks[2], stride=2, group_norm=group_norm)),
            nn.Sequential(*_make_layer(block, res_base_width*8, num_blocks[3], stride=2, group_norm=group_norm)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        all_layers.append(local_layer)
    elif fedexnn_split_num == 3:
        all_layers = []
        local_layer = nn.Sequential(
            nn.Conv2d(in_channels, res_base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_base_width, group_norm),
            nn.ReLU(),
        )
        all_layers.append(local_layer)

        local_layer = nn.Sequential(
            nn.Sequential(*_make_layer(block, res_base_width, num_blocks[0], stride=1, group_norm=group_norm)),
            nn.Sequential(*_make_layer(block, res_base_width*2, num_blocks[1], stride=2, group_norm=group_norm)),
        )
        all_layers.append(local_layer)

        local_layer = nn.Sequential(
            nn.Sequential(*_make_layer(block, res_base_width*4, num_blocks[2], stride=2, group_norm=group_norm)),
            nn.Sequential(*_make_layer(block, res_base_width*8, num_blocks[3], stride=2, group_norm=group_norm)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        all_layers.append(local_layer)
    elif fedexnn_split_num == 4:
        all_layers = []
        local_layer = nn.Sequential(
            nn.Conv2d(in_channels, res_base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_base_width, group_norm),
            nn.ReLU(),
        )
        all_layers.append(local_layer)

        local_layer = nn.Sequential(
            nn.Sequential(*_make_layer(block, res_base_width, num_blocks[0], stride=1, group_norm=group_norm)),
            nn.Sequential(*_make_layer(block, res_base_width*2, num_blocks[1], stride=2, group_norm=group_norm)),
        )
        all_layers.append(local_layer)
        all_layers.append(nn.Sequential(*_make_layer(block, res_base_width*4, num_blocks[2], stride=2, group_norm=group_norm)))
        local_layer = nn.Sequential(
            nn.Sequential(*_make_layer(block, res_base_width*8, num_blocks[3], stride=2, group_norm=group_norm)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        all_layers.append(local_layer)
    elif fedexnn_split_num == 5:
        all_layers = []
        local_layer = nn.Sequential(
            nn.Conv2d(in_channels, res_base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_base_width, group_norm),
            nn.ReLU(),
        )
        all_layers.append(local_layer)
        all_layers.append(nn.Sequential(*_make_layer(block, res_base_width, num_blocks[0], stride=1, group_norm=group_norm)))
        all_layers.append(nn.Sequential(*_make_layer(block, res_base_width*2, num_blocks[1], stride=2, group_norm=group_norm)))
        all_layers.append(nn.Sequential(*_make_layer(block, res_base_width*4, num_blocks[2], stride=2, group_norm=group_norm)))
        local_layer = nn.Sequential(
            nn.Sequential(*_make_layer(block, res_base_width*8, num_blocks[3], stride=2, group_norm=group_norm)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        all_layers.append(local_layer)
    elif fedexnn_split_num > 6:
        all_layers = []
        local_layer = nn.Sequential(
            nn.Conv2d(in_channels, res_base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_base_width, group_norm),
            nn.ReLU(),
        )
        all_layers.append(local_layer)
        all_layers.append(nn.Sequential(*_make_layer(block, res_base_width, num_blocks[0], stride=1, group_norm=group_norm)))
        all_layers.append(nn.Sequential(*_make_layer(block, res_base_width*2, num_blocks[1], stride=2, group_norm=group_norm)))
        all_layers.append(nn.Sequential(*_make_layer(block, res_base_width*4, num_blocks[2], stride=2, group_norm=group_norm)))
        local_layer = nn.Sequential(
            nn.Sequential(*_make_layer(block, res_base_width*8, num_blocks[3], stride=2, group_norm=group_norm)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        all_layers.append(local_layer)
    else:
        raise NotImplementedError
    return all_layers



def define_fl_exnn_res_layers(
    block, num_blocks, group_norm=0, res_base_width=64, in_channels=3,
    hetero_layer_depth=False,
):
    in_planes = res_base_width
    group_norm = group_norm
    # in_planes = in_planes * block.expansion
    layers = []
    def _make_layer(block, planes, num_blocks, stride):
        nonlocal in_planes
        nonlocal layers
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(block(in_planes, planes, stride))
            in_planes = planes * block.expansion

    layers.append(
        nn.Sequential(nn.Conv2d(in_channels, res_base_width, kernel_size=3, stride=1, padding=1, bias=False),
                    norm2d(res_base_width, group_norm),
                    nn.ReLU())
            )
    _make_layer(block, res_base_width, num_blocks[0], stride=1)
    _make_layer(block, res_base_width*2, num_blocks[1], stride=2)
    _make_layer(block, res_base_width*4, num_blocks[2], stride=2)
    _make_layer(block, res_base_width*8, num_blocks[3], stride=2)
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    if hetero_layer_depth:
        normal_layers = deepcopy(layers)
        layers = []
        layers.append(
            nn.Sequential(nn.Conv2d(in_channels, res_base_width, kernel_size=3, stride=1, padding=1, bias=False),
                        norm2d(res_base_width, group_norm),
                        nn.ReLU())
                )
        _make_layer(block, res_base_width, num_blocks[0]-1, stride=1)
        _make_layer(block, res_base_width*2, num_blocks[1]-1, stride=2)
        _make_layer(block, res_base_width*4, num_blocks[2]-1, stride=2)
        _make_layer(block, res_base_width*8, num_blocks[3]-1, stride=2)
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        small_layers = deepcopy(layers)
        layers = []
        layers.append(
            nn.Sequential(nn.Conv2d(in_channels, res_base_width, kernel_size=3, stride=1, padding=1, bias=False),
                        norm2d(res_base_width, group_norm),
                        nn.ReLU())
                )
        _make_layer(block, res_base_width, num_blocks[0]+1, stride=1)
        _make_layer(block, res_base_width*2, num_blocks[1]+1, stride=2)
        _make_layer(block, res_base_width*4, num_blocks[2]+1, stride=2)
        _make_layer(block, res_base_width*8, num_blocks[3]+1, stride=2)
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        large_layers = deepcopy(layers)
    else:
        small_layers, large_layers = None, None
    return layers, small_layers, large_layers



def fl_exnn_resnet18(**kwargs):
    return define_fl_exnn_res_layers( 
                BasicBlock, [2, 2, 2, 2], **kwargs)


def fl_exnn_resnet34(**kwargs):
    return define_fl_exnn_res_layers(
                BasicBlock, [3, 4, 6, 3], **kwargs)


def fl_exnn_resnet50(**kwargs):
    return define_fl_exnn_res_layers(
                Bottleneck, [3, 4, 6, 3], **kwargs)


def fl_exnn_resnet101(**kwargs):
    return define_fl_exnn_res_layers(
                Bottleneck, [3, 4, 23, 3], **kwargs)


def fl_exnn_resnet152(**kwargs):
    return define_fl_exnn_res_layers(
                Bottleneck, [3, 8, 36, 3], **kwargs)











class Federated_EXNNLayer_local(nn.Module):

    def __init__(self, layer_idx,
                local_layer,
                client_idx=0,
                adapter="avg", 
                fedexnn_self_dropout=0.0
            ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_global = False
        self.local_layer = local_layer
        self.client_idx = client_idx
        # self.in_features = local_layer.in_features
        self.adapter = adapter
        self.fedexnn_self_dropout = fedexnn_self_dropout
        if self.fedexnn_self_dropout > 0:
            self.dropout = nn.Dropout(p=self.fedexnn_self_dropout)

    def adaptation(self, in_channels=1, out_channels=1):
        if self.adapter in ["avg", "sum"]:
            pass
        elif self.adapter == "cnn1x1":
            self.adapter_nn = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        elif self.adapter == "mlp":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def get_module(self):
        return self.local_layer

    def forward(self, x, is_global):
        if is_global:
            if self.fedexnn_self_dropout > 0:
                x[str(self.client_idx)] = self.dropout(x[str(self.client_idx)])
            if self.adapter == "avg":
                xs = list(x.values())
                x = sum(xs) / len(xs)
            elif self.adapter == "sum":
                x = sum(list(x.values()))
            elif self.adapter == "cnn1x1":
                # x = self.adapter_layers[str(i)](  torch.concat(list(x.values()), dim=1) )
                x = self.adapter_nn(torch.concat(list(x.values()), dim=1))

            elif self.adapter == "mlp":
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            if self.fedexnn_self_dropout > 0:
                x = self.dropout(x)
        return self.local_layer(x)








class Federated_EXNNLayer_global(nn.Module):

    def __init__(self, layer_idx,
                local_layers, fedexnn_self_dropout=0.0
            ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_global = True
        self.fedexnn_self_dropout = fedexnn_self_dropout

        self.local_layers = torch.nn.ModuleDict()
        for client_idx, local_layer in local_layers.items():
            self.local_layers[client_idx] = local_layer

        # logger.info(f"list(local_layers.values())[0]: {list(local_layers.values())[0]}")

        # if hasattr(list(local_layers.values())[0], "in_features"):
        #     self.in_features = list(local_layers.values())[0].in_features
        # else:
        #     pass

    def get_module(self):
        return self.local_layers

    def freeze(self):
        for client_idx, local_layer in self.local_layers.items():
            for param in local_layer.parameters():
                param.requires_grad = False


    def forward(self, x, is_global):
        xs = {}
        for client_idx, local_layer in self.local_layers.items():
            xs[client_idx] = local_layer(x, is_global)
        return xs


# def merge_layer(Federated_EXNNs, layer_idx):
#     horizon_layers = {}
#     # for client_idx, Federated_EXNN in enumerate(Federated_EXNNs):
#     for client_idx, Federated_EXNN in Federated_EXNNs.items():
#         logger.info(f"Merging layer {layer_idx}, model has num of layers:{len(Federated_EXNN.layers)}")
#         horizon_layers[client_idx] = Federated_EXNN.layers[layer_idx].get_module()
#     federated_EXNNLayer_global = Federated_EXNNLayer_global(layer_idx, horizon_layers)
#     return federated_EXNNLayer_global



def merge_layer(Federated_EXNNs, layer_idx):
    horizon_layers = {}
    # for client_idx, Federated_EXNN in enumerate(Federated_EXNNs):
    for client_idx, Federated_EXNN in Federated_EXNNs.items():
        logger.info(f"Merging layer {layer_idx}, model has num of layers:{len(Federated_EXNN.layers)}")
        horizon_layers[str(client_idx)] = Federated_EXNN.layers[layer_idx]
    federated_EXNNLayer_global = Federated_EXNNLayer_global(layer_idx, horizon_layers)
    return federated_EXNNLayer_global



class Federated_EXNN(nn.Module):
    def __init__(
        self,
        args,
        client_idx,
        split_local_layers=[],
        num_of_classes=10,
        fedexnn_classifer="avg",
    ):
        super().__init__()
        self.args = args
        self.client_idx = client_idx
        self.num_layers = len(split_local_layers)
        if args.model == "cnn":
            self.hidden_features = args.cnn_hidden_features * 4 * 4
            self.flatten_at_classifier = True
        elif args.model == "mlp3":
            self.hidden_features = args.mlp_hidden_features
            self.flatten_at_classifier = False
        elif args.model in ["resnet18",]:
            self.hidden_features = args.res_base_width * 8 * 1
            self.flatten_at_classifier = True
        elif args.model in ["resnet50", ]:
            self.hidden_features = args.res_base_width * 8 * 4
            self.flatten_at_classifier = True
        else:
            raise NotImplementedError
        self.num_of_classes = num_of_classes
        # self.adapter = adapter
        # self.adapter_layers = torch.nn.ModuleDict()

        self.layers = nn.ModuleList()
        for i, local_layer in enumerate(split_local_layers):
            # lay = Federated_EXNNLayer_local(
            #     i,
            #     local_layer,
            # )
            # self.layers.append(lay)
            self.layers.append(local_layer)

        self.classifier = torch.nn.Linear(self.hidden_features, num_of_classes)
        self.fedexnn_classifer = fedexnn_classifer

    # def adaptation(self, layer_idx, federated_EXNNLayer_global, in_channels=1, out_channels=1):
    def adaptation(self, layer_idx, federated_EXNNLayer_global):
        federated_EXNNLayer_global.freeze()
        del self.layers[layer_idx]
        self.layers.insert(layer_idx, federated_EXNNLayer_global)
        # self.layers[layer_idx] = federated_EXNNLayer_global

    def add_local_layer_adaptor(self, layer_idx, **kwargs):
        assert not self.layers[layer_idx].is_global
        self.layers[layer_idx].adaptation(**kwargs)

    def get_last_training_adapter(self):
        for layer_idx, layer in enumerate(self.layers):
            if not layer.is_global and hasattr(layer, "adapter_nn"):
                return layer.adapter_nn
        return None

    # def adaptation(self, layer_idx, federated_EXNNLayer_global, in_channels=1, out_channels=1):
    #     federated_EXNNLayer_global.freeze()
    #     # self.layers[layer_idx] = federated_EXNNLayer_global
    #     # self.layers.pop(layer_idx)
    #     del self.layers[layer_idx]
    #     self.layers.insert(layer_idx, federated_EXNNLayer_global)
    #     # self.layers[layer_idx] = federated_EXNNLayer_global
    #     if layer_idx < self.num_layers - 1:
    #         if self.adapter in ["avg", "sum"]:
    #             pass
    #         elif self.adapter == "cnn1x1":
    #             self.adapter_layers[str(layer_idx)] = nn.Conv2d(
    #                 in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    #         elif self.adapter == "mlp":
    #             raise NotImplementedError
    #         else:
    #             raise NotImplementedError
    #     else:
    #         pass


    def adaptation_classifier(self, fedexnn_classifer, new_classifier=None):
        self.fedexnn_classifer = fedexnn_classifer
        self.classifier = new_classifier


    def forward(self, x, get_logits=False):
        # x = x.contiguous()
        # if self.flatten_at_classifier:
        #     pass
        # else:
        #     x = x.contiguous()
        #     x = x.view(x.size(0), self.layers[0].in_features)
        #     # logger.info(f"type(x): {type(x)}")

        prev_is_global = False
        for i, lay in enumerate(self.layers):
            # first layer is raw data, no need to adapt
            x = lay(x, prev_is_global)
            prev_is_global = lay.is_global
            # if getattr(lay, "is_global", False) and i < self.num_layers - 1:

        logits = None
        if getattr(self.layers[-1], "is_global", False):
            if self.flatten_at_classifier:
                for k, v in x.items():
                    x[k] = v.view(v.size(0), -1)
            else:
                pass
            if self.fedexnn_classifer in ["avg"] :
                x = sum(list(x.values()))
                if get_logits:
                    logits = x
                outputs = self.classifier(x)
            elif self.fedexnn_classifer == "multihead":
                outputs = self.classifier(x)
                outputs = sum(x)
            else:
                raise NotImplementedError
        else:
            if self.flatten_at_classifier:
                x = x.view(x.size(0), -1)
                if get_logits:
                    logits = x
            else:
                pass
            outputs = self.classifier(x)
        if get_logits:
            return outputs, logits
        else:
            return outputs



    def forward_measure(self, x):
        hidden_xs = {}
        # for layer_index, module in enumerate(self._layers.values()):
        # for layer_index, module in enumerate(self._layers):
        prev_is_global = False
        for i, lay in enumerate(self.layers):
            x = lay(x, prev_is_global)
            prev_is_global = lay.is_global
            # The outputed x is globally, thus average it for measuring MI. 
            xs = list(x.values())
            x_avg = sum(xs) / len(xs)
            x_avg = x_avg.detach()
            hidden_xs[i] = x_avg
        current_i = i

        if getattr(self.layers[-1], "is_global", False):
            if self.flatten_at_classifier:
                for k, v in x.items():
                    x[k] = v.view(v.size(0), -1)
            else:
                pass
            if self.fedexnn_classifer in ["avg"] :
                x = sum(list(x.values()))
                outputs = self.classifier(x)
            elif self.fedexnn_classifer == "multihead":
                outputs = self.classifier(x)
                outputs = sum(x)
            else:
                raise NotImplementedError
        else:
            if self.flatten_at_classifier:
                x = x.view(x.size(0), -1)
            else:
                pass
            outputs = self.classifier(x)
        # hidden_xs[current_i+1] = outputs
        return outputs, hidden_xs



    def get_final_features(self, x):
        prev_is_global = False
        for i, lay in enumerate(self.layers):
            # first layer is raw data, no need to adapt
            x = lay(x, prev_is_global)
            prev_is_global = lay.is_global


        if getattr(self.layers[-1], "is_global", False):
            if self.flatten_at_classifier:
                for k, v in x.items():
                    x[k] = v.view(v.size(0), -1)
            else:
                pass
            if self.fedexnn_classifer in ["avg"] :
                x = sum(list(x.values()))
            elif self.fedexnn_classifer == "multihead":
                pass
            else:
                raise NotImplementedError
        else:
            pass

        return x


















