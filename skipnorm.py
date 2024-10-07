import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init

class SkipNorm(nn.Module):

    def __init__(self, normalized_shape, window_size, eps: float = 1e-5, 
                elementwise_affine=True, bias= True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.normalized_shape = (normalized_shape,)
        #self.window_size = (window_size,)
        self.window_size = window_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
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
        if len(self.skips) < self.window_size:
            self.skips.append(input)
        else:
            self.skips = self.skips[1:] + [input] 


    def forward(self, input):
        if len(self.skips) < self.window_size:
            self.add_skip(input)
            return input
        else:
            prev = torch.stack(self.skips)
            out = (input - torch.mean(prev)) / torch.sqrt(torch.std(prev)**2 + self.eps)
            out = self.weight * out + self.bias 
            
        self.add_skip(input)
        return out

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)