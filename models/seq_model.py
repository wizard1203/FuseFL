import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out


class Sequential_SplitNN(nn.Module):
    def __init__(self, split_train, split_config, split_measure_config, local_module_num,
                layers=[]):
        super(Sequential_SplitNN, self).__init__()
        self.split_train = split_train
        self.split_config = split_config
        self.split_measure_config = split_measure_config
        # self.class_num = class_num
        self.local_module_num = local_module_num
        self._layers = nn.ModuleList([])

        for layer in layers:
            self._layers.append(layer)

        if self.split_train:
            for module_index, layer_index in enumerate(self.split_config):
                pass
                # exec('self.decoder_' + str(module_index) + 
                #     '= Decoder(wide_list[module_index], image_size, widen=aux_net_widen)')

                # exec('self.aux_classifier_' + str(module_index) +
                #     '= AuxClassifier(wide_list[module_index], net_config=aux_net_config, '
                #     'loss_mode=local_loss_mode, class_num=class_num, '
                #     'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')

    def add_layer(self, layer):
        self._layers.append(layer)



    def forward(self, *x, get_logits=False):
        # for layer in self._layers.values():
        logits = None
        for layer_index, layer in enumerate(self._layers):
            if isinstance(x, tuple):
                x = layer(*x)
            else:
                x = layer(x)
            if layer_index == len(self._layers) - 2:
                logits = x

        if get_logits:
            return x, logits
        else:
            return x



    def forward_measure(self, x):
        # local_module_i = 0
        # hidden_xs = []
        hidden_xs = {}
        # for layer_index, module in enumerate(self._layers.values()):
        for layer_index, module in enumerate(self._layers):
            if isinstance(x, tuple):
                x = module(*x)
            else:
                x = module(x)
            # if self.split_measure_config[local_module_i] == layer_index:
            if layer_index in self.split_measure_config:
                x = x.detach()
                hidden_xs[layer_index] = x
                # elif layer_index == len(self._layers) - 1:
                #     x = x.detach()
                #     hidden_xs[layer_index] = x

        # if not self.split_measure_config[-1] == layer_index:
        #     hidden_xs[layer_index] = x

        return x, hidden_xs


        # local_module_i = 0
        # hidden_xs = []
        # hidden_channels = []
        # for layer_idx, layer in enumerate(self.layers):
        #     x = layer(x)
        #     if local_module_i < 16:
        #         if self.MI_split_config[local_module_i] == layer_idx:
        #             x = x.detach()
        #             hidden_xs.append(x)
        #             local_module_i += 1

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # logits = self.linear(x)
        # logits = logits.detach()
        # hidden_xs.append(logits)
        # return logits, hidden_xs


class LinearProbes(nn.Module):
    def __init__(self):
        super(LinearProbes, self).__init__()
        self.probes = torch.nn.ModuleDict()
        self.criterion = nn.CrossEntropyLoss()

    def add(self, layer_index, hidden_features, num_of_classes):
        self.probes[str(layer_index)] = torch.nn.Linear(hidden_features, num_of_classes)

    def forward(self, img, hidden_xs, target):
        loss_ixys = []
        h_logits = []
        bs = img.shape[0]
        for layer_index, x in hidden_xs.items():
            x = x.detach()
            x = x.view(bs, -1)
            logits = self.probes[str(layer_index)](x)
            loss_ixy = self.criterion(logits, target)
            h_logits.append(logits)
            loss_ixy.backward()
            loss_ixys.append(loss_ixy.item())

        return h_logits, loss_ixys



    




class ReconMIEstimator(nn.Module):
    def __init__(self, split_measure_config):
        super(ReconMIEstimator, self).__init__()
        self.split_measure_config = split_measure_config
        self.decoders = torch.nn.ModuleDict()
        self.aux_classifiers = torch.nn.ModuleDict()
        self.criterion = nn.CrossEntropyLoss()


    def add_decoder(self, decoder, layer_index):
        self.decoders[str(layer_index)] = decoder

    def add_aux_classifier(self, aux_classifier, layer_index):
        self.aux_classifiers[str(layer_index)] = aux_classifier


    def forward(self, img, hidden_xs, target):
        loss_ixx_modules = []
        loss_ixy_modules = []
        h_logits = []

        # img_restore = self._image_restore(img)
        img_restore = img
        # for layer_idx, layer in enumerate(self.layers):
        # for module_index, layer_index in enumerate(self.infopro_config):
        for layer_index, x in hidden_xs.items():
            # x = hidden_xs[module_index]
            # print(f"x.shape:{x.shape}, img_restore.shape: {img_restore.shape}")
            # loss_ixx = eval('self.decoder_' + str(module_index))(x, img_restore)
            # logging.info(f"hidden xs layer:{layer_index}, ")
            # logging.info(f"==================================")
            # logger.info(f"hidden xs layer:{layer_index}, ")
            # logger.info(f"==================================")
            # print(f"==================================")
            x = x.detach()
            loss_ixx = self.decoders[str(layer_index)](x, img_restore)
            image_size = img.shape[-1]
            x = F.interpolate(x, size=[image_size, image_size],
                        mode='bilinear', align_corners=True)

            # logits, loss_ixy = eval('self.aux_classifier_' + str(module_index))(x, target)
            logits = self.aux_classifiers[str(layer_index)](x)
            loss_ixy = self.criterion(logits, target)

            h_logits.append(logits)
            loss_ixx.backward()
            loss_ixy.backward()
            loss_ixx_modules.append(loss_ixx.item())
            loss_ixy_modules.append(loss_ixy.item())

        return h_logits, loss_ixx_modules, loss_ixy_modules







































