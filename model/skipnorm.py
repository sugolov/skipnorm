import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init


# helper functions
subset_window = lambda a, l, w: a[max(0,l-w):l]

class SkipNorm(nn.Module):

    def __init__(self, dim, depth, window, eps: float = 1e-5, 
                elementwise_affine=True, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.dim = dim
        self.window = window
        self.depth = depth
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty((self.depth, self.dim), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty((self.depth, self.dim), **factory_kwargs))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.skips = self._reset_skips()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def reset(self):
        self.skips = self._reset_skips()

    def _reset_skips(self):
        #return torch.zeros(self.window_size + self.normalized_shape)
        return []

    def add_skip(self, input):
        self.skips.append(input)
        #if len(self.skips) < self.window_size:
        #    self.skips.append(input)
        #else:
        #    self.skips = self.skips[1:] + [input] 


    def forward(self, input, l):
        if l > 0:
            prev = subset_window(self.skips, l, self.window)
            prev = torch.stack(prev)

            out = (input - torch.mean(prev)) / torch.sqrt(torch.std(prev)**2 + self.eps)
            out = self.weight[l] * out + self.bias[l]
        else:
            out = input
            
        self.add_skip(input)
        return out

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)