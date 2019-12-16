import collections

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage.filters import threshold_otsu

import torch
import torch.nn as nn


class DenseNet(nn.Module):
    def __init__(self, in_dim, hidden_dims, dense=False, residual=False, nonlinearity=nn.LeakyReLU()):
        super(DenseNet, self).__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.nonlinearity = nonlinearity
        self.dense = dense
        self.residual = residual
        assert not (self.dense and self.residual)
        if self.dense:
            self.layers = nn.ModuleList()
            for i in range(self.num_layers):
                if i > 0:
                    in_dim = in_dim + self.hidden_dims[i-1]
                self.layers.append(nn.Linear(in_dim, hidden_dims[i], bias=True))
        else:
            self.layers = nn.ModuleList([nn.Linear(self.in_dim if i==0 else hidden_dims[i-1], hidden_dims[i], bias=True) 
                                        for i in range(self.num_layers)])
    
    def forward(self, x, return_layers='last'):
        xs = [x]
        for i in range(self.num_layers):
            if self.dense:
                x = torch.cat(xs, dim=1)
            x = self.layers[i](x)
            if self.residual and i > 0:
                x = x + xs[-1]
            if i < self.num_layers - 1:
                x = self.nonlinearity(x)
            xs.append(x)
        if return_layers=='last':
            return x
        if return_layers=='all':
            return xs
        if isinstance(return_layers, (list, tuple)):
            return [xs[i] for i in return_layers]


class MultiConv(nn.Module):
    r"""Applies two convolutions over an input of signal composed of several input planes.
    
    Args:
        in_channels (int): Number of input channels
    
    Shape:
        Input: :math:`(N, in_channels, H, W)`
        Output: :math:`(N, out_channels, H_{out}, W_{out})`
    
    Attributes:
        weight (Tensor): 
        bias (Tensor): 
        
    Examples::
    
        >>> x = torch.randn(2, 3, 5, 7)
        >>> model = MultiConv(3, 11)
        >>> model(x).shape
    
    """
    def __init__(self, in_channels, out_channels, num_conv=2, n_dim=2, kernel_size=3, padding=1, padding_mode='replicate', same_shape=True, 
                 normalization='layer_norm', activation=nn.LeakyReLU(negative_slope=0.01, inplace=True)):
        super(MultiConv, self).__init__()
        self.num_conv = num_conv
        if n_dim == 3:
            Conv = nn.Conv3d
        elif n_dim == 2:
            Conv = nn.Conv2d
        elif n_dim == 1:
            Conv = nn.Conv1d
        # self.padding needs to be an Iterable here if same_shape is False and padding_mode is 'reflect' or 'replicate'
        self.padding = padding if isinstance(padding, collections.abc.Iterable) else [padding]*n_dim
        self.padding_mode = padding_mode
        assert padding_mode in ['zeros', 'circular', 'reflect', 'replicate'], f'padding_mode={padding_mode} not handled!'
        if same_shape:
            if padding_mode == 'zeros':
                if isinstance(kernel_size, int):
                    assert kernel_size % 2 == 1, f'Arguments kernel_size and padding cannot maintain the input shape!'
                    self.padding = (kernel_size - 1)//2
                elif isinstance(kernel_size, collections.abc.Iterable):
                    self.padding = []
                    for k in kernel_size:
                        assert k % 2 == 1, f'Arguments kernel_size and padding cannot maintain the input shape!'
                        self.padding.append((k-1)//2)
                else:
                    raise ValueError(f'Argument kernel_size = {kernel_size} not handled!')
            else: 
                # based on pytorch 1.1.0 'reflect', 'replicate', and 'circular' padding behave differently from 'zeros'
                # this may change later
                if isinstance(kernel_size, int):
                    self.padding = [kernel_size - 1] * n_dim
                elif isinstance(kernel_size, collections.abc.Iterable):
                    self.padding = [k-1 for k in kernel_size]
                else:
                    raise ValueError(f'Argument kernel_size = {kernel_size} not handled!')
        self.conv = nn.ModuleList()
        self.norm = nn.ModuleList()
        for i in range(self.num_conv):
            self.conv.append(Conv(in_channels if i==0 else out_channels, out_channels, kernel_size, 
                                  padding=0 if self.padding_mode in ['reflect', 'replicate'] else self.padding,
                                  padding_mode=self.padding_mode))
            if normalization == 'layer_norm':
                num_groups = 1
            elif normalization == 'instance_norm':
                num_groups = out_channels
            elif isinstance(normalization, int):
                num_groups = normalization
            else:
                raise ValueError(f'normalization = {normalization} not defined!')
            self.norm.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_channels))
        self.activation = activation
        
    def forward(self, x):
        for i in range(self.num_conv):
            if self.padding_mode=='reflect' or self.padding_mode=='replicate':
                expanded_padding = []
                for p in self.padding:
                    expanded_padding += [(p+1)//2, p//2]
                x = nn.functional.pad(x, expanded_padding, mode=self.padding_mode)
            x = self.activation(self.norm[i](self.conv[i](x)))
        return x

    
class DownConv(nn.Module):
    r"""
    Args:
    Shape:
    Attributes:
    Examples:
    """
    def __init__(self, in_channels, out_channels, num_conv=2, n_dim=2, kernel_size=3, padding=1, padding_mode='replicate', same_shape=True, 
                 normalization='layer_norm', activation=nn.LeakyReLU(negative_slope=0.01, inplace=True)):
        super(DownConv, self).__init__()
        self.n_dim = n_dim
        if self.n_dim==3:
            Maxpool = nn.MaxPool3d
        elif self.n_dim==2:
            Maxpool = nn.MaxPool2d
        elif self.n_dim==1:
            Maxpool = nn.MaxPool1d
        self.downconv = nn.Sequential(Maxpool(kernel_size=2), 
                                      MultiConv(in_channels, out_channels, num_conv=num_conv, n_dim=n_dim, kernel_size=kernel_size, padding=padding,
                                                padding_mode=padding_mode, same_shape=same_shape, normalization=normalization, activation=activation))
    
    def forward(self, x):
        # handle the case for maxpool when size[i] == 1 (dimension i has length 1); maxpool will output 0 size and raise error
        size = x.size()
        pad = []
        for s in reversed(size[-self.n_dim:]):
            if s == 1:
                pad = pad + [0, 1]
            else:
                pad = pad + [0, 0]
        if sum(pad) > 0:
            x = nn.functional.pad(x, pad, mode='replicate')
        return self.downconv(x)

    
class UpConv(nn.Module):
    r"""
    Args:
    Shape:
    Attributes:
    Examples:
    """
    def __init__(self, in_channels, out_channels, num_conv=2, n_dim=2, kernel_size=3, padding=1, padding_mode='replicate', same_shape=True, 
                 normalization='layer_norm', activation=nn.LeakyReLU(negative_slope=0.01, inplace=True)):
        super(UpConv, self).__init__()
        self.n_dim = n_dim
        self.padding_mode = padding_mode
        if self.n_dim==3:
            ConvTranspose = nn.ConvTranspose3d
        elif self.n_dim==2:
            ConvTranspose = nn.ConvTranspose2d
        elif self.n_dim==1:
            ConvTranspose = nn.ConvTranspose1d
        self.up = ConvTranspose(in_channels//2, in_channels//2, kernel_size=2, stride=2) # parameterizable
        self.conv = MultiConv(in_channels, out_channels, num_conv=num_conv, n_dim=self.n_dim, kernel_size=kernel_size, padding=padding, 
                              padding_mode=padding_mode, same_shape=same_shape, normalization=normalization, activation=activation)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dw = x2.size(-1) - x1.size(-1)
        pad = [dw//2, dw-dw//2]
        if self.n_dim > 1:
            dh = x2.size(-2) - x1.size(-2)
            pad = pad + [dh//2, dh-dh//2]
        if self.n_dim > 2:
            dd = x2.size(-3) - x1.size(-3)
            pad = pad + [dd//2, dd - dd//2]
        x1 = nn.functional.pad(x1, pad, mode='constant' if self.padding_mode=='zeros' else self.padding_mode)
        return self.conv(torch.cat([x1, x2], dim=1))

    
class UNet(nn.Module):
    r"""
    Args:
    
    Shape:
    
    Attributes:
    
    Examples:
        n_dim = 3
        out_channels = [64, 128, 256, 512]
        in_channels = 3
        num_classes = 2
        if n_dim == 3:
            out_channels = [n//2 for n in out_channels]
        model = UNet(in_channels=in_channels, num_classes=num_classes, out_channels=out_channels, n_dim=n_dim)
        param_cnt = {n: p.numel() for n, p in model.named_parameters()}
        print(sum(param_cnt.values()))
        x = torch.randn(1, in_channels, 50, 100, 100)
        model(x).shape
    """
    def __init__(self, in_channels, num_classes, out_channels=[64, 128, 256, 512], num_conv=2, n_dim=2, kernel_size=3, padding=1, 
                 padding_mode='replicate', same_shape=True, normalization='layer_norm', activation=nn.LeakyReLU(negative_slope=0.01, inplace=True)):
        super(UNet, self).__init__()
        self.ndim = n_dim
        self.conv = MultiConv(in_channels, out_channels[0], num_conv=num_conv, n_dim=n_dim, kernel_size=kernel_size, padding=padding, 
                              padding_mode=padding_mode, same_shape=same_shape, normalization=normalization, activation=activation)
        self.encoder = nn.Sequential(collections.OrderedDict(
            [(f'down{i}', 
              DownConv(out_channels[i], out_channels[i+1] if i < len(out_channels)-1 else out_channels[-1], num_conv=num_conv, n_dim=n_dim, 
                       kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, same_shape=same_shape, normalization=normalization, 
                       activation=activation)) 
             for i in range(len(out_channels))]))
        self.decoder = nn.Sequential(collections.OrderedDict(
            [(f'up{i}', 
              UpConv(out_channels[i]*2, out_channels[i-1] if i > 0 else out_channels[0], num_conv=num_conv, n_dim=n_dim, kernel_size=kernel_size, 
                     padding=padding, padding_mode=padding_mode, same_shape=same_shape, normalization=normalization, activation=activation))
             for i in reversed(range(len(out_channels)))]))
        if n_dim==3:
            Conv = nn.Conv3d
        elif n_dim==2:
            Conv = nn.Conv2d
        elif n_dim==1:
            Conv = nn.Conv1d
        self.out = Conv(out_channels[0], num_classes, 1)
        
    def forward(self, x):
        add_ndim = self.ndim + 2 - x.ndim
        for i in range(add_ndim):
            x = x.unsqueeze(0)
        x = self.conv(x)
        xs = [x]
        for m in self.encoder:
            x = m(x)
            xs.append(x)
        for i, m in enumerate(self.decoder):
            x = m(x, xs[-i-2])
        x = self.out(x)
        for i in range(add_ndim):
            x = x.squeeze(0)
        return x

    
def get_mask(size, kernel_size, start, value=1, device=torch.device('cuda')):
    """Get a mask array or a list of mask arrays;
    A mask array has most zero elements and only a few elements are assigned to the given value
    
    Args:
        size: the shape of the mask array
        kernel_size: a list or tuple, assert len(kernel_size) == len(size)
        start: to return one mask, assert len(start) == len(size);
            to return multiple masks, assert np.array(start).ndim == 2 and np.array(start).shape[1] == len(size)
            to return all possible (np.prod(size)) masks, set start = None
        value: given value to assign to zero array, default 1
        
    Returns:
        one mask array or a list of mask arrays
        
    """
    if start is None:
        return [get_mask(size, kernel_size, start=np.unravel_index(i, kernel_size), value=value, device=device)
                for i in range(np.product(kernel_size))]        
    if np.array(start).ndim == 1:
        mask = torch.zeros(size, device=device)
        ndim = len(size)
        indices = []
        for i, (s, e, step) in enumerate(zip(start, size, kernel_size)):
            index = np.array(range(s, e, step))
            for j in range(ndim-i-1):
                index = np.expand_dims(index, -1)
            indices.append(index)
        mask[indices] = value
        return mask
    if np.array(start).ndim == 2:
        return [get_mask(size, kernel_size, start=s, value=value, device=device)
                for s in start]  
    
    
def get_bg_mat(mat, kernel_size, ndim=None, padding_mode='zeros', device=torch.device('cuda')):
    r"""Replace each element using its neighbors excluding itself
    
    Args:
        mat: 1-d, 2-d or 3-d tensor if ndim is None else mat can have an additional dim corresponding to batch size
        kernel_size: int or list (tuple) of ints
        ndim: if none, infer from mat, deciding the dim of conv filters; default none
        padding_mode: currently unused, only 'zeros' is handled
        device: use gpu when possible
        
    Returns:
        bg_mat: complementary to mat
        
    """
    if ndim is None:
        ndim = mat.dim()
    size = mat.size()
    for i in range(2+ndim-mat.dim()): # The first two dimensions should correspond to batches and channels 
        mat = mat.unsqueeze(0)
    assert mat.size(1) == 1 # only handles the input channel == 1
    if ndim == 1:
        Conv = nn.Conv1d
    elif ndim == 2:
        Conv = nn.Conv2d
    elif ndim == 3:
        Conv = nn.Conv3d
    else:
        raise ValueError(f'kernel_size with {len(kernel_size)} > 3 elements has not been handled')
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * ndim
    for k in kernel_size:
        assert k%2!=0, f'kernel_size should be odd number (>=3) to maintain the same shape!'
    assert len(kernel_size) == ndim
    if padding_mode == 'zeros':
        padding = [k//2 for k in kernel_size]
    else:
        raise ValueError(f'padding_mode={padding_mode} is not handled')
    model = Conv(1, 1, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, bias=False).to(device)
    for n, p in model.named_parameters():
        p.requires_grad_(False)
        p.fill_(1)
        p[tuple([0, 0] + padding)] = 0 # do not include self
#     model.weight.data = 1-get_mask(kernel_size, kernel_size, start=padding).view(model.weight.size()) # another way to set fixed weight
    torch.cuda.empty_cache()
    with torch.no_grad():
        bg_mat = model(mat).squeeze() / model(torch.ones(mat.size(), device=device))
    torch.cuda.empty_cache()
    return bg_mat.view(size)


def restore_image_noise2self(model, x, bg_masks=None, kernel_size=3, bg_mat=None, is_train=False):
    y_pred = 0
    if bg_masks is None:
        size = x.shape[2:]
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]*len(size)
        masks = get_mask(size, kernel_size, start=None)
        bg_masks = [1-mask for mask in masks]
    with torch.set_grad_enabled(is_train):
        for bg_mask in bg_masks:
            input = x*bg_mask
            if bg_mat is not None:
                input += bg_mat * (1 - bg_mask)
            y_pred = y_pred + model(input) * (1-bg_mask)
    return y_pred