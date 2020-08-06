import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from utility import get_label_image, get_tensor_slice, get_prime_factors
from optical_electrophysiology import extract_traces
from models import UNet

def get_traces(mat, softmask, label_image=None, regions=None, min_pixels=20, percentile=50, label_size_max_portion=0.5, min_thresh=0.1):
    if label_image is None:
        label_image, regions = get_label_image(softmask, min_pixels=min_pixels, label_size_max_portion=label_size_max_portion, min_thresh=min_thresh)
    submats, traces = extract_traces(mat, softmask=softmask, label_image=label_image, regions=regions, percentile=percentile)
    return label_image, traces
    
def get_high_conf_mask(cor_map, low_percentile=25, high_percentile=5, min_cor=0.1, min_pixels=20, exclude_boundary_width=2):
    label_image, regions = get_label_image(cor_map, min_pixels=min_pixels)
    fg_mask = ((cor_map >= np.percentile(cor_map.cpu(), high_percentile)) & (cor_map.new_tensor(label_image)>0) & (cor_map > min_cor))
    bg_mask = ((cor_map <= np.percentile(cor_map.cpu(), low_percentile)) & (cor_map.new_tensor(label_image)==0) & (cor_map < min_cor))
    mask = cor_map.new_full(cor_map.size(), -1)
    mask[bg_mask] = 0
    mask[fg_mask] = 1
    mask[:exclude_boundary_width] = -1
    mask[-exclude_boundary_width:] = -1
    mask[:, :exclude_boundary_width] = -1
    mask[:, -exclude_boundary_width:] = -1
    return mask

def semi_supervised_segmentation(mat, cor_map=None, model=None, out_channels=[8,16,32], kernel_size=3, frames_per_iter=100, 
                                 num_iters=200, print_every=1, select_frames=False, return_model=False,
                                 optimizer_fn=torch.optim.AdamW, optimizer_fn_args = {'lr': 1e-2, 'weight_decay': 1e-3}, 
                                 loss_threshold=0, save_loss_folder=None, reduction='max', last_out_channels=None,
                                 verbose=False, device=torch.device('cuda')):
    if cor_map is None:
        cor_map = get_cor_map_4d(mat, select_frames=select_frames)
    high_conf_mask = get_high_conf_mask(cor_map)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([(high_conf_mask==1).sum(), (high_conf_mask==0).sum()]).float().to(device))
    loss_history = []
    if model is None:
#         nrow, ncol = mat.shape[-2:]
#         pool_kernel_size_row = get_prime_factors(nrow)[:3]
#         pool_kernel_size_col = get_prime_factors(ncol)[:3]
#         model = UNet(in_channels=mat.shape[0], num_classes=2, out_channels=out_channels, num_conv=2, n_dim=3, 
#                      kernel_size=[3, (3, 3, 3), (3, 3, 3), (3, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)], 
#                      padding=[1, (0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 1)], 
#                      pool_kernel_size=[(2, pool_kernel_size_row[0], pool_kernel_size_col[0]), 
#                                        (2, pool_kernel_size_row[1], pool_kernel_size_col[1]), 
#                                        (2, pool_kernel_size_row[2], pool_kernel_size_col[2])], 
#                      use_adaptive_pooling=True, same_shape=False,
#                      transpose_kernel_size=[(1, pool_kernel_size_row[2], pool_kernel_size_col[2]), 
#                                             (1, pool_kernel_size_row[1], pool_kernel_size_col[1]), 
#                                             (1, pool_kernel_size_row[0], pool_kernel_size_col[0])], 
#                      transpose_stride=[(1, pool_kernel_size_row[2], pool_kernel_size_col[2]), 
#                                        (1, pool_kernel_size_row[1], pool_kernel_size_col[1]), 
#                                        (1, pool_kernel_size_row[0], pool_kernel_size_col[0])],
#                      padding_mode='zeros', normalization='layer_norm',
#                      activation=nn.LeakyReLU(negative_slope=0.01, inplace=True)).to(device)
        if mat.ndim == 3:
            mat = mat.unsqueeze(0)
        in_channels = mat.shape[0]
        num_classes = 2
        if isinstance(kernel_size, int):
            padding = (kernel_size-1)//2
            padding_row = padding_col = padding
        elif isinstance(kernel_size, tuple):
            assert len(kernel_size)==3
            padding, padding_row, padding_col = [(k-1)//2 for k in kernel_size]
        encoder_depth = len(out_channels)
        nframe, nrow, ncol = mat.shape[-3:]
        if isinstance(kernel_size, int):
            assert nframe > 4*encoder_depth*(kernel_size-1)
        else:
            assert nframe > 4*encoder_depth*(kernel_size[0]-1)
        pool_kernel_size_row = get_prime_factors(nrow)[:encoder_depth]
        pool_kernel_size_col = get_prime_factors(ncol)[:encoder_depth]
        model = UNet(in_channels=in_channels, num_classes=num_classes, out_channels=out_channels, num_conv=2, n_dim=3, 
                     kernel_size=kernel_size, 
                     padding=[padding] + [(0, padding, padding)]*encoder_depth*2, 
                     pool_kernel_size=[(1, pool_kernel_size_row[i], pool_kernel_size_col[i]) for i in range(encoder_depth)], 
                     use_adaptive_pooling=True, same_shape=False,
                     transpose_kernel_size=[(1, pool_kernel_size_row[i], pool_kernel_size_col[i]) 
                                            for i in reversed(range(encoder_depth))], 
                     transpose_stride=[(1, pool_kernel_size_row[i], pool_kernel_size_col[i]) 
                                       for i in reversed(range(encoder_depth))],
                     padding_mode='zeros', normalization='layer_norm', last_out_channels=last_out_channels,
                     activation=nn.LeakyReLU(negative_slope=0.01, inplace=True)).to(device)
    optimizer = optimizer_fn(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_fn_args)
    
    idx = torch.nonzero(high_conf_mask!=-1, as_tuple=True)
    y_true = high_conf_mask[idx].long()
    for i in range(num_iters):
#         if (i+1) == num_iters//2:
#             optimizer_fn_args['lr'] /= 10
#             optimizer = optimizer_fn(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_fn_args)
        x = get_tensor_slice(mat, dims=[1], sizes=[frames_per_iter])
        y_pred = model(x)
        if reduction == 'max':
            y_pred = y_pred.max(1)[0]
        elif reduction == 'mean':
            y_pred = y_pred.mean(1)
        elif reduction.startswith('top'):
            n = y_pred.size(1)
            if reduction.endswith('percent'):
                k = max(int(int(reduction[3:-7])/100. * n), 1)
            else:
                k = min(int(reduction[3:]), n)
            y_pred = y_pred.topk(k, dim=1)[0].mean(1)
        else:
            raise ValueError(f'reduction = {reduction} not handled!')
        y_pred = y_pred[:, idx[0], idx[1]].T
        loss = loss_fn(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if verbose and ((i+1)%print_every == 0 or i==0 or i==num_iters-1):
            print(f'{i+1} loss={loss.item()}')
        if loss_threshold>0 and (i+1)%print_every==0 and np.mean(loss_history[-print_every:])<loss_threshold:
            break
    if verbose:
        plt.title('Training loss')
        plt.plot(loss_history, 'ro-', markersize=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()
    if save_loss_folder is not None and os.path.exists(save_loss_folder):
        np.save(f'{save_loss_folder}/loss__semi_supervised_segmentation.npy', loss_history)
    with torch.no_grad():
        soft_masks = []
        num_iters = mat.shape[1] // frames_per_iter + 1
        for i in range(num_iters):
            x = get_tensor_slice(mat, dims=[1], sizes=[frames_per_iter])
            y_pred = model(x)
            if reduction == 'max':
                y_pred = y_pred.max(1)[0]
            elif reduction == 'mean':
                y_pred = y_pred.mean(1)
            elif reduction.startswith('top'):
                n = y_pred.size(1)
                if reduction.endswith('percent'):
                    k = max(int(int(reduction[3:-7])/100. * n), 1)
                else:
                    k = min(int(reduction[3:]), n)
                y_pred = y_pred.topk(k, dim=1)[0].mean(1)
            else:
                raise ValueError(f'reduction = {reduction} not handled!')
            soft_mask = torch.softmax(y_pred, dim=0)[1]
            soft_masks.append(soft_mask)
        soft_mask = torch.stack(soft_masks, dim=0).mean(0)
    if return_model:
        return soft_mask, model 
    else:
        return soft_mask