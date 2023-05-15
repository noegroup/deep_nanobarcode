import torch
import torch.nn as nn


cuda_0 = torch.device('cuda:0')
cpu = torch.device('cpu')

if torch.cuda.is_available():
    nn_device = cuda_0
else:
    nn_device = cpu


class CompoundActivationBlock(nn.Module):

    def __init__(self,
                 use_dropout=True, use_bn=True,
                 dropout_rate=0.5, num_features=None):

        super(CompoundActivationBlock, self).__init__()

        self.activation = nn.ReLU()

        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()

        if use_bn:
            self.batch_norm = nn.BatchNorm1d(num_features)
        else:
            self.batch_norm = nn.Identity()

    def forward(self, x, **kwargs):

        return self.dropout(self.activation(self.batch_norm(x, **kwargs), **kwargs), **kwargs)


class ResidualBlock(nn.Module):

    def __init__(self, block_depth, use_bias=False,
                 num_features=None, **activation_kwargs):

        super(ResidualBlock, self).__init__()

        self.identity = nn.Identity()

        self.layer = nn.ModuleList()
        self.activation = nn.ModuleList()

        for i in range(block_depth):
            self.layer.append(nn.Linear(num_features, num_features, bias=use_bias))

            self.activation.append(CompoundActivationBlock(num_features=num_features, **activation_kwargs))

    def forward(self, x, **kwargs):

        y = self.identity(x)

        for _lay, _act in zip(self.layer[:-1], self.activation[:-1]):
            x = _act(_lay(x, **kwargs), **kwargs)

        z = self.activation[-1](self.layer[-1](x, **kwargs) + y, **kwargs)

        return z
