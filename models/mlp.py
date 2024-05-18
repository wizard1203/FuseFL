import torch
import torch.nn.functional as F
from torch import nn


from .configs import InfoPro, InfoPro_balanced_memory
from .basics import View


class MLPMIEstimator(nn.Module):
    def __init__(self, args, arch, local_module_num, in_features, hidden_features, num_of_classes, num_layers,
                batch_size, image_size, dataset='cifar10',
                wide_list=(16, 16, 32, 64), dropout_rate=0,
                aux_net_config='1c2f', local_loss_mode='contrast',
                aux_net_widen=1, aux_net_feature_dim=128):
        super().__init__()
        self.args = args

        self.num_layers = num_layers
        self.infopro_config = InfoPro[arch][local_module_num]

        for item in self.infopro_config:
            module_index, layer_index = item
            exec('self.decoder_' + str(module_index) + '_' + str(layer_index) +
                 '= Decoder(wide_list[module_index], image_size, widen=aux_net_widen)')
            exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) +
                 f'= org_resnet32(input_channels={hidden_x_channels[local_module_i]})')
            local_module_i += 1

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

# class MLP(nn.Module):
#     def __init__(self, in_features, hidden_features, num_of_classes, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         self.layers = nn.ModuleList()
#         self.in_features = in_features
#         layer = nn.Linear(in_features, hidden_features)
#         self.layers.append(layer)
#         for _ in range(self.num_layers - 1):
#             layer = nn.Linear(hidden_features, hidden_features)
#             self.layers.append(layer)
#         self.classifier = torch.nn.Linear(hidden_features, num_of_classes)


#     def forward(self, x):
#         x = x.contiguous()
#         x = x.view(x.size(0), self.in_features)
#         for lay in self.layers:
#             # x = [F.relu(el) for el in lay(x)]
#             x = F.relu(lay(x))
#         outputs = self.classifier(x)
#         return outputs



class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, num_of_classes, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.in_features = in_features
        for _ in range(self.num_layers):
            # layer = nn.Linear(hidden_features, hidden_features)
            self.layers.append(nn.Sequential(nn.Linear(hidden_features, hidden_features),
                            nn.ReLU())
                        )
        self.classifier = torch.nn.Linear(hidden_features, num_of_classes)


    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self.in_features)
        for lay in self.layers:
            x = lay(x)
        outputs = self.classifier(x)
        return outputs


    def forward_measure_MI(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self.in_features)

        hidden_xs = []
        hidden_channels = []
        local_module_i = 0
        for lay in self.layers:
            x = lay(x)
            x = x.detach()
            hidden_xs.append(x)
            local_module_i += 1

        outputs = self.classifier(x)

        return outputs, hidden_xs




def make_MLP_seqs(in_features, hidden_features, out_features, init_classifier, num_layers):
    layers = []
    layers.append(nn.Sequential(
                    View([in_features]),
                    nn.Linear(in_features, hidden_features),
                    nn.ReLU())
                  )
    for _ in range(num_layers - 1):
        layers.append(nn.Sequential(nn.Linear(hidden_features, hidden_features),
                    nn.ReLU())
                )
    if init_classifier:
        classifier = torch.nn.Linear(hidden_features, out_features)
        layers.append(classifier)

    return layers


def make_MLP_Head_seqs(in_features, hidden_features, out_features, init_classifier, split_layer_index, num_layers):
    origin_res_layer_index = 0
    layers = []
    if origin_res_layer_index > split_layer_index:
        layers.append(nn.Sequential(nn.Linear(in_features, hidden_features),
                    nn.ReLU())
                )
    origin_res_layer_index += 1

    for _ in range(num_layers - 1):
        if origin_res_layer_index > split_layer_index:
            layers.append(nn.Sequential(nn.Linear(hidden_features, hidden_features),
                        nn.ReLU())
                    )
        origin_res_layer_index += 1

    if init_classifier:
        if origin_res_layer_index > split_layer_index:
            classifier = torch.nn.Linear(hidden_features, out_features)
            layers.append(classifier)
        origin_res_layer_index += 1
    return layers





def mlp2(in_features, hidden_features, out_features, init_classifier):
    return make_MLP_seqs(in_features, hidden_features, out_features, init_classifier, 2)


def mlp3(in_features, hidden_features, out_features, init_classifier):
    return make_MLP_seqs(in_features, hidden_features, out_features, init_classifier, 3)





























