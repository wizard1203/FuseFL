from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


from .group_normalization import GroupNorm2d
from .configs import InfoPro
from .auxiliary_nets import Decoder, AuxClassifier
from .basics import View
from .seq_model import Sequential_SplitNN


def norm2d(planes, num_channels_per_group=32):
    print("num_channels_per_group:{}".format(num_channels_per_group))
    if num_channels_per_group > 0:
        return GroupNorm2d(
            planes, num_channels_per_group, affine=True, track_running_stats=False
        )
    else:
        return nn.BatchNorm2d(planes)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = norm2d(planes, group_norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = norm2d(planes, group_norm)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, group_norm=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = norm2d(planes, group_norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = norm2d(planes, group_norm)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.bn3 = norm2d(planes * self.expansion, group_norm)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, group_norm=0, res_base_width=64, in_channels=3):
        super(ResNet, self).__init__()
        # self.in_planes = 64
        self.res_base_width = res_base_width
        self.in_planes = self.res_base_width

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.conv1 = nn.Conv2d(in_channels, self.res_base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm2d(self.res_base_width, group_norm)
        self.layer1 = self._make_layer(block, self.res_base_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.res_base_width*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.res_base_width*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.res_base_width*8, num_blocks[3], stride=2)
        self.linear = nn.Linear(self.res_base_width * 8 * block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        out = F.relu(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)

        if return_features:
            return out, feature
        else:
            return out


class ResNetMIEstimator(nn.Module):
    def __init__(self, args, arch, hidden_x_channels, local_module_num, batch_size, image_size=32,
                 balanced_memory=False, dataset='cifar10', class_num=10,
                 wide_list=(16, 16, 32, 64), 
                 aux_net_config='1c2f', local_loss_mode='contrast',
                 aux_net_widen=1, aux_net_feature_dim=128):
        super(ResNetMIEstimator, self).__init__()
        self.args = args

        self.inplanes = wide_list[0]
        self.feature_num = wide_list[-1]
        self.class_num = class_num
        self.local_module_num = local_module_num

        self.image_size = image_size

        self.infopro_config = InfoPro[arch][local_module_num]
        for module_index, layer_index in enumerate(self.infopro_config):
            exec('self.decoder_' + str(module_index) + 
                 '= Decoder(wide_list[module_index], image_size, widen=aux_net_widen)')
            exec('self.aux_classifier_' + str(module_index) +
                 f'= resnet18_head(self.args, split_layer_index=layer_index)')

        if 'cifar' in dataset:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [125.3, 123.0, 113.9]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [63.0, 62.1, 66.7]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()
        else:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()


    def _image_restore(self, normalized_image):
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) \
               + self.mask_train_mean[:normalized_image.size(0)]


    def forward(self, img, hidden_xs, target):
        loss_ixx_modules = []
        loss_ixy_modules = []
        h_logits = []

        img_restore = self._image_restore(img)
        # for layer_idx, layer in enumerate(self.layers):
        for module_index, layer_index in enumerate(self.infopro_config):
            if self.infopro_config[module_index] == module_index:
                x = hidden_xs[module_index]
                # print(f"x.shape:{x.shape}, img_restore.shape: {img_restore.shape}")
                loss_ixx = eval('self.decoder_' + str(module_index))(x, img_restore)
                x = F.interpolate(x, size=[self.image_size, self.image_size],
                            mode='bilinear', align_corners=True)

                logits, loss_ixy = eval('self.aux_classifier_' + str(module_index))(x, target)
                h_logits.append(logits)
                loss_ixx.backward()
                loss_ixy.backward()
                loss_ixx_modules.append(loss_ixx.item())
                loss_ixy_modules.append(loss_ixy.item())
                x = x.detach()

        return h_logits, loss_ixx_modules, loss_ixy_modules



# def make_ResNetMIEstimator(layers, split_measure_config, hidden_x_channels, image_size,
#                         aux_net_widen=1):
def make_ResNetMIEstimator(layers, hidden_x_channels, image_size,
                        aux_net_widen=1):
    decoders = {}
    aux_classifiers = {}
    # wide_list = 

    # for module_index, layer_index in enumerate(split_measure_config):
    for layer_index, x_channel in hidden_x_channels.items():
        # exec('self.decoder_' + str(module_index) + 
        #         '= Decoder(wide_list[module_index], image_size, widen=aux_net_widen)')
        # exec('self.aux_classifier_' + str(module_index) +
        #         f'= resnet18_head(self.args, split_layer_index=layer_index)')
        decoders[layer_index] = Decoder(
            x_channel, image_size, widen=aux_net_widen)
        sub_layers = deepcopy(layers)[layer_index+1:]
        aux_classifiers[layer_index] = Sequential_SplitNN(False, None, 
            split_measure_config=None, local_module_num=None, layers=sub_layers)
        # aux_classifiers[layer_index] = Aux_Classifier(
        #     True, split_layer_index=layer_index)
    return decoders, aux_classifiers





def get_res18_out_channels(res_base_width):
    out_channels = []
    num_blocks = [2, 2, 2, 2]
    in_planes = res_base_width

    def _make_layer(planes, num_blocks, stride):
        nonlocal in_planes
        nonlocal out_channels
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            out_channels.append(planes)
            in_planes = planes
    out_channels.append(res_base_width)
    _make_layer(res_base_width, num_blocks[0], stride=1)
    _make_layer(res_base_width*2, num_blocks[1], stride=2)
    _make_layer(res_base_width*4, num_blocks[2], stride=2)
    _make_layer(res_base_width*8, num_blocks[3], stride=2)
    return out_channels




def make_ResNet_seqs(init_classifier, 
            block, num_blocks, num_classes=10, group_norm=0, res_base_width=64, in_channels=3):
    # in_planes = 64
    in_planes = res_base_width
    # layers = nn.ModuleList([])
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
    # linear = nn.Linear(res_base_width * 8 * block.expansion, num_classes)
    # torch.nn.AvgPool2d(kernel_size, stride=None, padding=0)
    layers.append(
        nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                        View([-1]))
                    )
    layers.append(nn.Linear(res_base_width * 8 * block.expansion, num_classes))
    return layers


# def forward(self, x, return_features=False):

#     for layer in self.layers:
#         x = layer(x)

#     out = F.adaptive_avg_pool2d(x, (1, 1))
#     feature = out.view(out.size(0), -1)
#     out = self.linear(feature)
#     if return_features:
#         return out, feature
#     else:
#         return out


# def forward_measure_MI(self, x):
#     local_module_i = 0
#     hidden_xs = []
#     hidden_channels = []
#     for layer_idx, layer in enumerate(self.layers):
#         x = layer(x)
#         if local_module_i < 16:
#             if self.MI_split_config[local_module_i] == layer_idx:
#                 x = x.detach()
#                 hidden_xs.append(x)
#                 local_module_i += 1

#     x = self.avgpool(x)
#     x = x.view(x.size(0), -1)
#     logits = self.linear(x)
#     logits = logits.detach()
#     hidden_xs.append(logits)
#     return logits, hidden_xs


def make_ResNet_Head_seqs(init_classifier, split_layer_index,  
            block, num_blocks, num_classes=10, group_norm=0, res_base_width=64, in_channels=3):
    # in_planes = 64
    # args = args
    origin_res_layer_index = 0
    in_planes = res_base_width
    layers = []

    def _make_layer(block, planes, num_blocks, stride):
        nonlocal in_planes
        nonlocal layers
        nonlocal origin_res_layer_index
        nonlocal split_layer_index
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            if origin_res_layer_index > split_layer_index:
                layers.append(block(in_planes, planes, stride))
                in_planes = planes * block.expansion
            origin_res_layer_index += 1

    if origin_res_layer_index > split_layer_index:
        layers.append(
            nn.Sequential(nn.Conv2d(in_channels, res_base_width, kernel_size=3, stride=1, padding=1, bias=False),
                        norm2d(res_base_width, group_norm),
                        nn.ReLU())
                )
    origin_res_layer_index += 1

    _make_layer(block, res_base_width, num_blocks[0], stride=1)
    _make_layer(block, res_base_width*2, num_blocks[1], stride=2)
    _make_layer(block, res_base_width*4, num_blocks[2], stride=2)
    _make_layer(block, res_base_width*8, num_blocks[3], stride=2)
    if origin_res_layer_index > split_layer_index:
        layers.append(
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                            View([-1]))
                        )
    origin_res_layer_index += 1

    if init_classifier:
        if origin_res_layer_index > split_layer_index:
            layers.append(nn.Linear(res_base_width * 8 * block.expansion, num_classes))
        origin_res_layer_index += 1

    return layers



# class ResNet_Head(nn.Module):
#     def __init__(self, args, arch, split_layer_index, 
#             block, num_blocks, num_classes=10, group_norm=0, res_base_width=64, in_channels=3):
#         super(ResNet_Head, self).__init__()
#         # self.in_planes = 64
#         self.res_base_width = res_base_width
#         self.in_planes = self.res_base_width
#         self.split_layer_index = split_layer_index
#         self.origin_res_layer_index = 0

#         self.layers = nn.ModuleList([])

#         self.layers.append(
#             nn.Sequential(nn.Conv2d(in_channels, self.res_base_width, kernel_size=3, stride=1, padding=1, bias=False),
#                         norm2d(self.res_base_width, group_norm),
#                         nn.ReLU())
#                 )

#         self._make_layer(block, self.res_base_width, num_blocks[0], stride=1)
#         self._make_layer(block, self.res_base_width*2, num_blocks[1], stride=2)
#         self._make_layer(block, self.res_base_width*4, num_blocks[2], stride=2)
#         self._make_layer(block, self.res_base_width*8, num_blocks[3], stride=2)
#         self.linear = nn.Linear(self.res_base_width * 8 * block.expansion, num_classes)
#         self.layers.append(
#             nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
#                            View([-1]))
#                         )
#         self.layers.append(nn.Linear(self.res_base_width * 8 * block.expansion, num_classes))


#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         for stride in strides:
#             if self.origin_res_layer_index > self.split_layer_index:
#                 self.layers.append(block(self.in_planes, planes, stride))
#                 self.in_planes = planes * block.expansion
#             self.origin_res_layer_index += 1
#         # return nn.Sequential(*layers)


#     def forward(self, x, return_features=False):

#         for layer in self.layers:
#             x = layer(x)

#         out = F.adaptive_avg_pool2d(x, (1, 1))
#         feature = out.view(out.size(0), -1)
#         out = self.linear(feature)
#         if return_features:
#             return out, feature
#         else:
#             return out




def resnet18_layers(init_classifier, num_classes=10, **kwargs):
    return make_ResNet_seqs(init_classifier, BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)


def resnet34_layers(init_classifier, num_classes=10, **kwargs):
    return make_ResNet_seqs(init_classifier, BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)

def resnet50_layers(init_classifier, num_classes=10, **kwargs):
    return make_ResNet_seqs(init_classifier, Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)


def resnet18_head(init_classifier, split_layer_index, **kwargs):
    return make_ResNet_Head_seqs(init_classifier, split_layer_index, BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet50_head(init_classifier, split_layer_index, **kwargs):
    return make_ResNet_Head_seqs(init_classifier, split_layer_index, Bottleneck, [3, 4, 6, 3], **kwargs)


# def resnet18_layers(args, local_module_num, num_classes=10, **kwargs):
#     return make_ResNet_seqs(args, "resne18", local_module_num, BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)


# def resnet34_layers(args, local_module_num, num_classes=10, **kwargs):
#     return make_ResNet_seqs(args, "resne18", local_module_num, BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)


# def resnet18_head(args, split_layer_index, **kwargs):
#     model = ResNet_Head(args, "resne18", split_layer_index, BasicBlock, [5, 5, 5], arch='resnet32', **kwargs)
#     return model




def resnet18(args, local_module_num, num_classes=10, **kwargs):
    return ResNet(args, "resne18", local_module_num, BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)


def resnet34(args, local_module_num, num_classes=10, **kwargs):
    return ResNet(args, "resne18", local_module_num, BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)


def resnet50(args, local_module_num, num_classes=10, **kwargs):
    return ResNet(args, "resne18", local_module_num, Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)


def resnet101(num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)


def resnet152(num_classes=10, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, **kwargs)











