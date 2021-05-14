import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import sigmoid
from torch.jit import script
from torch.autograd import Function
from torch.nn import ReLU, LeakyReLU
from torch.nn import Module, ModuleList, Sequential, Conv1d, BatchNorm1d, Dropout, ConvTranspose1d
from datetime import datetime

@script
def swish_jit_fwd(x):
    return x * sigmoid(x)


@script
def swish_jit_bwd(x, grad):
    x_s = sigmoid(x)
    return grad * (x_s * (1 + x * (1 - x_s)))


class SwishAutoFn(Function):

    @staticmethod
    def symbolic(g, x):
        return g.op('Swish', x)

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad)


class Swish(Module):
    """
    Swish Activation function
    https://arxiv.org/abs/1710.05941
    """
    def forward(self, x):
        return SwishAutoFn.apply(x)


activations = {
    "relu": ReLU,
    "swish": Swish,
}


class Model(Module):
    """
    Model template for QuartzNet style architectures
    https://arxiv.org/pdf/1910.10261.pdf
    """
    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config
        self.stride = config['block'][0]['stride'][0]
        self.alphabet = config['labels']['labels']
        self.features = config['block'][-1]['filters']
        self.encoder = Encoder(config)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        if beamsize == 1 or qscores:
            seq, path  = viterbi_search(x, self.alphabet, qscores, self.qscale, self.qbias)
        else:
            seq, path = beam_search(x, self.alphabet, beamsize, threshold)
        if return_path: return seq, path
        return seq


class Encoder(Module):
    """
    Builds the model encoder
    """
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        features = self.config['input']['features']
        activation = activations[self.config['encoder']['activation']]()
        encoder_layers = []

        for layer in self.config['block']:
            if "type" in layer and layer["type"] == 'BlockX': 
                encoder_layers.append(
                    BlockX(
                        features, layer['filters'], layer['pool'], layer['inner_size'], activation,
                        repeat=layer['repeat'], kernel_size=layer['kernel'],
                        stride=layer['stride'], dilation=layer['dilation'],
                        dropout=layer['dropout'], residual=layer['residual'],
                        separable=layer['separable'],
                    )
                )
            elif "type" in layer and layer["type"] == 'dynpool':
                encoder_layers.append(
                    BlockDP(
                        features, layer['filters'], activation,
                        kernel_size=layer['kernel'], dropout=layer['dropout'],
                        prediction_size=layer['predictor_size'],
                        norm=layer['norm']
                    )
                )
            else:
                encoder_layers.append(
                    Block(
                        features, layer['filters'], activation,
                        repeat=layer['repeat'], kernel_size=layer['kernel'],
                        stride=layer['stride'], dilation=layer['dilation'],
                        dropout=layer['dropout'], residual=layer['residual'],
                        separable=layer['separable'],
                    )
                )

            features = layer['filters']

        self.encoder = Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder(x)


class TCSConv1d(Module):
    """
    Time-Channel Separable 1D Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, separable=False):

        super(TCSConv1d, self).__init__()
        self.separable = separable

        if separable:
            self.depthwise = Conv1d(
                in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias, groups=in_channels
            )

            self.pointwise = Conv1d(
                in_channels, out_channels, kernel_size=1, stride=1,
                dilation=dilation, bias=bias, padding=0
            )
        else:
            self.conv = Conv1d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=bias
            )

    def forward(self, x):
        if self.separable:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)
        return x

def shift(x, k):
    return torch.cat([torch.zeros(k).to(x.device).to(x.dtype), x[:-k]])

class BlockDP(Module):
    def __init__(self, in_channels, out_channels, activation, kernel_size, norm=None, prediction_size=32, dropout=0.05):
        super(BlockDP, self).__init__()

        self.conv = ModuleList()
        self.conv.extend(
            self.get_tcs(
                in_channels, out_channels, kernel_size=kernel_size,
                padding=kernel_size[0]//2, separable=False
            )
        )


        self.predictor = torch.nn.Sequential(
            torch.nn.Conv1d(2, prediction_size, 31, stride=1, padding=15),
            torch.nn.BatchNorm1d(prediction_size),
            Swish(),
            torch.nn.Conv1d(prediction_size, prediction_size, 15, stride=1, padding=7),
            torch.nn.BatchNorm1d(prediction_size),
            Swish(),
            torch.nn.Conv1d(prediction_size, 2, 15, stride=1, padding=7)
        )

        self.predictor[-1].weight.data *= 0.01
        self.predictor[-1].bias.data *= 0
        self.predictor[-1].bias.data[0] -= np.log(2)
        self.predictor[-1].bias.data[1] -= np.log(2)
        self.activation = Sequential(*self.get_activation(activation, dropout))
        self.norm_target = norm

        self.register_buffer('norm_mean', torch.ones(1)) 

    def get_activation(self, activation, dropout):
        return nn.GLU(dim=1), Dropout(p=dropout)

    def row_pool(self, features, moves, weights):
        fw = features * weights.to(features.dtype).unsqueeze(1)
        
        
        poses = torch.cumsum(moves.detach(), 0)
        
        poses = poses.unsqueeze(1)
        
        floors = torch.floor(poses)
        ceils = floors + 1
        
        w1 = (1 - (poses - floors)).to(features.dtype)
        w2 = (1 - (ceils - poses)).to(features.dtype)

        out = torch.zeros((int(ceils[-1].item())+1, features.shape[1])).to(features.device).to(features.dtype)
       
        out.index_add_(0, floors.to(torch.long).squeeze(1), w1*fw)
        out.index_add_(0, ceils.to(torch.long).squeeze(1), w2*fw)
            
        return out

    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels*2, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            BatchNorm1d(out_channels*2, eps=1e-3, momentum=0.1)
        ]

    def forward(self, x):
        _x = x
        for layer in self.conv:
            _x = layer(_x)
        _x = self.activation(_x)
        features = _x
        jumps_mat = self.predictor(torch.cat([x, x*x], dim=1))
        weights = torch.sigmoid(jumps_mat[:,0,:])
        moves = torch.sigmoid(jumps_mat[:,1,:])
        bmoves = moves
        if self.training:
            renorm = (1 / self.norm_target / moves.mean().detach()).detach()
            self.norm_mean.copy_(0.99 * self.norm_mean + 0.01 * renorm)
        else:
            renorm = self.norm_mean

    
        moves = moves * renorm
    
        
        features = features.permute((0,2,1))
        lens = []
        x_evs = []
        for f, m, w in zip(features.unbind(0), moves.unbind(0), weights.unbind(0)):
            pooled = self.row_pool(f, m, w)
            x_evs.append(pooled)
            lens.append(pooled.shape[0])
        
        x_evs = torch.nn.utils.rnn.pad_sequence(x_evs, True)
        x_evs = x_evs.permute(0, 2, 1)
        x_evs = F.pad(x_evs, (0, 3 - (x_evs.shape[2] % 3)))
        #x_evs = self.activation(x_evs)

        return x_evs, lens, bmoves, weights

class SF(Module):
    def __init__(self):
        super(SF, self).__init__()

    def forward(self, x):
        a, b = x.chunk(2, dim=1) 
        ap = nn.functional.pad(a, (1,0))[:,:,:-1]
        bp = nn.functional.pad(b, (0,1))[:,:,1:]
        return torch.cat([ap, bp], dim=1)

class BlockX(Module):
    """
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, in_channels, out_channels, pool, inner_size, activation, repeat=5, kernel_size=1, stride=1, dilation=1, dropout=0.0, residual=False, separable=False):

        super(BlockX, self).__init__()

        self.use_res = residual
        self.conv = ModuleList()

        _in_channels = in_channels
        padding = self.get_padding(kernel_size[0], stride[0], dilation[0])

        self.conv.extend([
            SF(),
            Conv1d(_in_channels, inner_size*2, kernel_size=pool, stride=pool, padding=0, bias=False),
            Conv1d(inner_size*2, inner_size*2, kernel_size, padding=padding, bias=False, groups=inner_size*2),
            BatchNorm1d(inner_size*2, eps=1e-3, momentum=0.1),
            ])
        self.conv.extend(self.get_activation(activation, dropout))

        _in_channels = inner_size

        # add the first n - 1 convolutions + activation
        for _ in range(repeat - 2):
            self.conv.extend(
                self.get_tcs(
                    _in_channels, inner_size, kernel_size=kernel_size,
                    stride=stride, dilation=dilation,
                    padding=padding, separable=separable
                )
            )

            self.conv.extend(self.get_activation(activation, dropout))
            _in_channels = inner_size

        # add the last conv and batch norm
        self.conv.extend([
            Conv1d(inner_size, inner_size, kernel_size, padding=padding, bias=False, groups=inner_size),
            ConvTranspose1d(inner_size, out_channels*2, kernel_size=pool, stride=pool, padding=0, bias=False),
            BatchNorm1d(out_channels*2, eps=1e-3, momentum=0.1),
            ])

        # add the residual connection
        if self.use_res:
            self.residual = Sequential(*self.get_tcs(in_channels, out_channels))

        # add the activation and dropout
        self.activation = Sequential(*self.get_activation(activation, dropout))

    def get_activation(self, activation, dropout):
        return nn.GLU(dim=1), Dropout(p=dropout)

    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation

    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels*2, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            BatchNorm1d(out_channels*2, eps=1e-3, momentum=0.1)
        ]

    def forward(self, x):
        _x = x
        for layer in self.conv:
            _x = layer(_x)
        if self.use_res:
            _x += self.residual(x)
        return self.activation(_x)

class Block(Module):
    """
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, in_channels, out_channels, activation, repeat=5, kernel_size=1, stride=1, dilation=1, dropout=0.0, residual=False, separable=False):

        super(Block, self).__init__()

        self.use_res = residual
        self.conv = ModuleList()

        _in_channels = in_channels
        padding = self.get_padding(kernel_size[0], stride[0], dilation[0])

        # add the first n - 1 convolutions + activation
        for _ in range(repeat - 1):
            self.conv.extend(
                self.get_tcs(
                    _in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, dilation=dilation,
                    padding=padding, separable=separable
                )
            )

            self.conv.extend(self.get_activation(activation, dropout))
            _in_channels = out_channels

        # add the last conv and batch norm
        self.conv.extend(
            self.get_tcs(
                _in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, dilation=dilation,
                padding=padding, separable=separable
            )
        )

        # add the residual connection
        if self.use_res:
            self.residual = Sequential(*self.get_tcs(in_channels, out_channels))

        # add the activation and dropout
        self.activation = Sequential(*self.get_activation(activation, dropout))

    def get_activation(self, activation, dropout):
        return nn.GLU(dim=1), Dropout(p=dropout)

    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation

    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels*2, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            BatchNorm1d(out_channels*2, eps=1e-3, momentum=0.1)
        ]

    def forward(self, x):
        _x = x
        for layer in self.conv:
            _x = layer(_x)
        if self.use_res:
            _x += self.residual(x)
        return self.activation(_x)
