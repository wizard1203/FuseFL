import torch
import torch.nn.functional as F
from torch import nn




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



class LinearAdapter(nn.Module):
    def __init__(self, in_features, out_features_per_column, num_prev_modules):
        super().__init__()
        # Eq. 1 - lateral connections
        # one layer for each previous column. Empty for the first task.
        self.lat_layers = nn.ModuleList([])
        for _ in range(num_prev_modules):
            m = nn.Linear(in_features, out_features_per_column)
            self.lat_layers.append(m)

    def forward(self, x):
        assert len(x) == self.num_prev_modules
        hs = []
        for ii, lat in enumerate(self.lat_layers):
            hs.append(lat(x[ii]))
        return sum(hs)


class MLPAdapter(nn.Module):

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

        # Eq. 2 - MLP adapter. Not needed for the first task.
        self.V = nn.Linear(in_features * num_prev_modules, out_features_per_column)
        self.alphas = nn.Parameter(torch.randn(num_prev_modules))
        self.U = nn.Linear(out_features_per_column, out_features_per_column)

    def forward(self, x):
        if self.num_prev_modules == 0:
            return 0  # first adapter is empty

        assert len(x) == self.num_prev_modules
        assert len(x[0].shape) == 2, (
            "Inputs to MLPAdapter should have two dimensions: "
            "<batch_size, num_features>."
        )
        for i, el in enumerate(x):
            x[i] = self.alphas[i] * el
        x = torch.cat(x, dim=1)
        x = self.U(self.activation(self.V(x)))
        return x


class PNNColumn(nn.Module):
    def __init__(
        self,
        in_features,
        out_features_per_column,
        num_prev_modules,
        adapter="mlp",
    ):
        """
        :param in_features: size of each input sample
        :param out_features_per_column:
            size of each output sample (single column)
        :param num_prev_modules: number of previous columns
        :param adapter: adapter type. One of {'linear', 'mlp'} (default='mlp')
        """
        super().__init__()
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column
        self.num_prev_modules = num_prev_modules

        if adapter == "linear":
            self.adapter = LinearAdapter(
                in_features, out_features_per_column, num_prev_modules
            )
        elif adapter == "mlp":
            self.adapter = MLPAdapter(
                in_features, out_features_per_column, num_prev_modules
            )
        else:
            raise ValueError("`adapter` must be one of: {'mlp', `linear'}.")
        self.itoh = nn.Linear(in_features, out_features_per_column)


    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        prev_xs, last_x = x[:-1], x[-1]
        hs = self.adapter(prev_xs)
        hs += self.itoh(last_x)
        return hs


class Federated_PNNLayer(nn.Module):

    def __init__(self, in_features, out_features_per_column, adapter="mlp"):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column
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
            PNNColumn(
                self.in_features,
                self.out_features_per_column,
                self.num_columns,
                adapter=self.adapter,
            )
        )

    def forward(self, x):
        hs = []
        for ii in range(self.num_columns):
            hs.append(self.columns[ii](x[: ii + 1]))
        return hs


class Federated_PNN(nn.Module):

    def __init__(
        self,
        num_layers=1,
        in_features=784,
        hidden_features_per_column=100,
        num_of_classes=10,
        adapter="mlp",
        classifier_name="progressive",
    ):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.in_features = in_features
        self.out_features_per_columns = hidden_features_per_column
        self.num_of_classes = num_of_classes

        self.layers = nn.ModuleList()
        self.layers.append(Federated_PNNLayer(in_features, hidden_features_per_column))
        for _ in range(num_layers - 1):
            lay = Federated_PNNLayer(
                hidden_features_per_column,
                hidden_features_per_column,
                adapter=adapter,
            )
            self.layers.append(lay)

        self.classifier_name = classifier_name
        if self.classifier_name == "fixed":
            self.classifier = torch.nn.Linear(hidden_features_per_column, num_of_classes)
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
            self.classifier.adaptation(self.out_features_per_columns, self.num_of_classes)
        else:
            raise RuntimeError


    def forward(self, x):
        """Forward.
        """
        x = x.contiguous()
        x = x.view(x.size(0), self.in_features)

        num_columns = self.layers[0].num_columns

        x = [x for _ in range(num_columns)]
        for lay in self.layers:
            x = [F.relu(el) for el in lay(x)]

        if self.classifier_name == "fixed":
            x = sum(x)
            outputs = self.classifier(x)
        elif self.classifier_name == "progressive":
            outputs = self.classifier(x)
        else:
            raise NotImplementedError
        return outputs
















