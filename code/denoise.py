import os, time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from visualization import make_video_ffmpeg
from utility import get_prime_factors
from models import UNet, MultiConv

class SeparateNet(nn.Module):
    def __init__(self, in_channels=1, num_features=32, spatial_out_channels=[64, 64, 32],
                 num_conv=2, kernel_size=3, padding=1, pool_kernel_size=2, transpose_kernel_size=2, transpose_stride=2, 
                 last_out_channels=None, use_adaptive_pooling=False, padding_mode='replicate', 
                 temporal_out_channels=[32, 64, 32], temporal_kernel_size=(3, 1, 1),
                 normalization='layer_norm', activation=nn.LeakyReLU(negative_slope=0.01, inplace=True)):
        super(SeparateNet, self).__init__()
        self.in_channels = in_channels
        self.spatial_2d = UNet(in_channels=in_channels, num_classes=num_features, out_channels=spatial_out_channels, 
                          num_conv=num_conv, n_dim=2, kernel_size=kernel_size, padding=padding, 
                          pool_kernel_size=pool_kernel_size, 
                          transpose_kernel_size=transpose_kernel_size, 
                          transpose_stride=transpose_stride, 
                          last_out_channels=last_out_channels, 
                          use_adaptive_pooling=use_adaptive_pooling, same_shape=True, padding_mode=padding_mode, 
                          normalization=normalization,
                          activation=activation)

        self.temporal_1d = MultiConv(in_channels=num_features, out_channels=temporal_out_channels, num_conv=None, n_dim=3, 
                                kernel_size=temporal_kernel_size, padding=0, padding_mode='zeros', 
                                normalization=normalization, activation=activation, same_shape=False, last_layer_activation=False)
        
    def forward(self, x, spatial_only=False):
        nrow, ncol = x.shape[-2:]
        if x.ndim==4 and x.size(1)!=self.in_channels:
            batch_size, t = x.shape[:2]
            x = x.view(-1, nrow, ncol).unsqueeze(1)
            spatial_features = self.spatial_2d(x).view(batch_size, t, -1, nrow, ncol).transpose(1, 2)
        else:
            spatial_features = self.spatial_2d(x)
        if spatial_only:
            return spatial_features
        y = self.temporal_1d(spatial_features)
        if y.size(2)==1:
            y = y.squeeze(2)
        return y
    
    def forward_temporal(self, spatial_features):
        y = self.temporal_1d(spatial_features)
        if y.size(2)==1:
            y = y.squeeze(2)
        return y

def get_noise2self_train_data(mat, ndim=3, batch_size=5, frame_depth=6, mask_prob=0.05, features=None, frame_weight=None, frame_indices=None, 
                              return_frame_indices=False, window_size_row=None, window_size_col=None, weight=None):
    """
    Args:
        frame_depth: 2*frame_depth+1 is number of frames used to denoise the middle frame
        
    Examples:
        nframe, nrow, ncol = 100, 300, 200
        frame_depth = 6
        batch_size = 5
        mask_prob = 0.05
        mat = torch.randn(nframe, nrow, ncol)
        frame_indices = np.random.choice(range(frame_depth, nframe-frame_depth), batch_size)
        x, y_true, mask = get_noise2self_train_data(mat, batch_size=batch_size, frame_depth=frame_depth, mask_prob=mask_prob, 
                                                    frame_indices=frame_indices)
        mask = mask.float()
        tmp = torch.stack([mat[i-frame_depth:i+frame_depth+1] for i in frame_indices], dim=0)
        assert torch.norm(x[:, :frame_depth] - tmp[:, :frame_depth])==0 and torch.norm(x[:, frame_depth+1:] - tmp[:, frame_depth+1:])==0
        assert torch.norm(x[:, frame_depth]*(1-mask) - tmp[:, frame_depth]*(1-mask)) == 0
        assert torch.norm(x[:, frame_depth]*(1-mask) - tmp[:, frame_depth]*(1-mask)) == 0
        print((x[:, frame_depth][mask.bool()] - tmp[:, frame_depth][mask.bool()]!=0).float().mean().item())
        print(mat.mean().item(), tmp[:, frame_depth][mask.bool()].mean().item(), 
            mat.std().item(), tmp[:, frame_depth][mask.bool()].std().item())
    """
    nframe, nrow, ncol = mat.shape[0], mat.shape[-2], mat.shape[-1]
    mean = mat.mean()
    std = mat.std()
    if frame_indices is None:
        if frame_weight is not None:
            frame_weight = frame_weight[range(frame_depth, nframe-frame_depth)]
            frame_weight /= frame_weight.sum()
        frame_indices = np.random.choice(range(frame_depth, nframe-frame_depth), batch_size, p=frame_weight)
    y_true = mat[frame_indices]
    x = torch.stack([(mat[i-frame_depth:i+frame_depth+1] if features is None else torch.cat([mat[i-frame_depth:i+frame_depth+1], features], dim=0))
                     for i in frame_indices], dim=0)
    if weight is not None and weight.ndim == mat.ndim:
        weight = weight[frame_indices]
    if window_size_row is not None and window_size_col is not None:
        row_indices = np.random.choice(nrow-window_size_row, batch_size)
        col_indices = np.random.choice(ncol-window_size_col, batch_size)
        y_true = torch.stack([y_true[i, ..., slice(r, r+window_size_row), slice(c, c+window_size_col)] 
                              for i, (r, c) in enumerate(zip(row_indices, col_indices))], dim=0)
        x = torch.stack([x[i, ..., slice(r, r+window_size_row), slice(c, c+window_size_col)] 
                         for i, (r, c) in enumerate(zip(row_indices, col_indices))], dim=0)
        if weight is not None:
            if weight.ndim == mat.ndim:
                weight = torch.stack([weight[i, ..., slice(r, r+window_size_row), slice(c, c+window_size_col)]
                                      for i, (r, c) in enumerate(zip(row_indices, col_indices))], dim=0)
            elif weight.ndim == mat.ndim-1:
                weight = torch.stack([weight[..., slice(r, r+window_size_row), slice(c, c+window_size_col)] 
                                      for r, c in zip(row_indices, col_indices)], dim=0)
            else:
                raise ValueError(f'weight.ndim = {weight.ndim} is not handled for mat.ndim = {mat.ndim}')
        nrow = window_size_row
        ncol = window_size_col
    mask = (torch.rand(batch_size, nrow, ncol, device=mat.device) <= mask_prob).float()
    x[:, frame_depth] = x[:, frame_depth] * (1-mask) + mask * (torch.randn(batch_size, nrow, ncol, device=mat.device)*std + mean)
    if ndim==3:
        x = x.unsqueeze(1)
    output_dict = {'x': x, 'y_true': y_true, 'mask': mask.bool()}
    if return_frame_indices:
        if window_size_row is not None and window_size_col is not None:
            frame_indices = (frame_indices, row_indices, col_indices)
        output_dict['frame_indices'] = frame_indices
    if weight is not None:
        output_dict['weight'] = weight
    return output_dict

def model_denoise(mat, model, batch_size=20, frame_depth=6, normalize=True, features=None, ndim=3, frame_indices=None, replicate_pad=True):
    """
    Args:
        mat: size (nframe, nrow, ncol)
    """
    if normalize:
        mean_mat = mat.mean()
        std_mat = mat.std()
        mat = (mat - mean_mat) / std_mat
    with torch.no_grad():
        if frame_indices is None:
            nframe = mat.shape[0]
            frame_indices = range(frame_depth, nframe-frame_depth, batch_size)
        y = []
        for start in frame_indices:
            x = torch.stack([(mat[i-frame_depth:i+frame_depth+1] if features is None else 
                              torch.cat([mat[i-frame_depth:i+frame_depth+1], features], dim=0))
                             for i in range(start, min(nframe-frame_depth, start+batch_size))], dim=0)
            if ndim==3:
                x = x.unsqueeze(1)
            y_ = model(x).squeeze(1)
            if ndim==3:
                y_ = y_.squeeze(1)
            y.append(y_)
            torch.cuda.empty_cache()
        y = torch.cat(y, dim=0)
        if replicate_pad:
            # to make y the same shape with mat
            y = torch.cat([torch.stack([y[0]]*frame_depth, dim=0), y, torch.stack([y[-1]]*frame_depth, dim=0)], dim=0)
        if normalize:
            y = y * std_mat + mean_mat
    return y


def get_denoised_mat(mat, model=None, out_channels=[64, 64, 64], ndim=3, num_epochs=12, num_iters=600, print_every=300, batch_size=2,
                     mask_prob=0.05, frame_depth=6, frame_weight=None, movie_start_idx=250, movie_end_idx=750, 
                     save_folder='.',save_intermediate_results=False, normalize=True, last_out_channels=None,
                     loss_reg_fn=nn.MSELoss(), loss_history=[], loss_threshold=0, optimizer_fn=torch.optim.AdamW, 
                     optimizer_fn_args={'lr': 1e-3, 'weight_decay': 1e-2}, lr_scheduler=None, batch_size_eval=10, kernel_size_unet=3,
                     features=None, fps=60,
                     window_size_row=None, window_size_col=None, weight=None, verbose=False, return_model=False, device=torch.device('cuda')):
    if normalize:
        mean_mat = mat.mean()
        std_mat = mat.std()
        mat = (mat - mean_mat) / std_mat
    else:
        mean_mat = 0
        std_mat = 1
    if model is None:
        encoder_depth = len(out_channels)
        kernel_size = kernel_size_unet
        assert kernel_size%2==1
        padding = (kernel_size-1)//2
        nrow, ncol = mat.shape[-2:]
        pool_kernel_size_row = get_prime_factors(nrow)[:encoder_depth]
        pool_kernel_size_col = get_prime_factors(ncol)[:encoder_depth]
        if ndim == 2:
            in_channels = frame_depth*2 + 1
            if features is not None:
                in_channels += features.shape[0]
            model = UNet(in_channels=in_channels, num_classes=1, out_channels=out_channels, num_conv=2, n_dim=2, 
                         kernel_size=kernel_size, 
                         padding=padding, 
                         pool_kernel_size=[(pool_kernel_size_row[i], pool_kernel_size_col[i]) for i in range(encoder_depth)], 
                         transpose_kernel_size=[(pool_kernel_size_row[i], pool_kernel_size_col[i]) for i in reversed(range(encoder_depth))], 
                         transpose_stride=[(pool_kernel_size_row[i], pool_kernel_size_col[i]) for i in reversed(range(encoder_depth))], 
                         last_out_channels=last_out_channels, 
                         use_adaptive_pooling=False, same_shape=True, padding_mode='replicate', normalization='layer_norm',
                         activation=nn.LeakyReLU(negative_slope=0.01, inplace=True)).to(device)
        elif ndim == 3:
            assert frame_depth % encoder_depth == 0
            k = frame_depth//encoder_depth + 1
            model = UNet(in_channels=1, num_classes=1, out_channels=out_channels, num_conv=2, n_dim=3, 
                         kernel_size=[kernel_size] + [(k, kernel_size, kernel_size)]*encoder_depth + [(1, kernel_size, kernel_size)]*encoder_depth, 
                         padding=[padding] + [(0, padding, padding)]*encoder_depth*2,
                         pool_kernel_size=[(1, pool_kernel_size_row[i], pool_kernel_size_col[i]) for i in range(encoder_depth)], 
                         transpose_kernel_size=[(1, pool_kernel_size_row[i], pool_kernel_size_col[i]) for i in reversed(range(encoder_depth))], 
                         transpose_stride=[(1, pool_kernel_size_row[i], pool_kernel_size_col[i]) for i in reversed(range(encoder_depth))], 
                         last_out_channels=last_out_channels,
                         use_adaptive_pooling=True, same_shape=False, padding_mode='zeros', normalization='layer_norm', 
                         activation=nn.LeakyReLU(negative_slope=0.01, inplace=True)).to(device)
    optimizer = optimizer_fn(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_fn_args)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if save_intermediate_results:
        movie_tyx = mat[movie_start_idx:movie_end_idx]*std_mat + mean_mat
        if not os.path.exists(f'{save_folder}/movie_frame{movie_start_idx}to{movie_end_idx}_raw.avi'):
            make_video_ffmpeg(movie_tyx, save_path=f'{save_folder}/movie_frame{movie_start_idx}to{movie_end_idx}_raw.avi', fps=fps)
    initial_start_time = time.time()
    for epoch in range(num_epochs):
        if lr_scheduler is not None and len(lr_scheduler)>=2:
            lr = lr_scheduler['lr_fn'](epoch, **lr_scheduler['lr_fn_args'])
            optimizer_fn_args['lr'] = lr
            optimizer = optimizer_fn(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_fn_args)
            if verbose:
                print(f'Epoch {epoch+1} set learning rate to be {lr:.2e}')
        start_time = time.time()
        for i in range(num_iters):
            batch_data = get_noise2self_train_data(mat, ndim=ndim, batch_size=batch_size, frame_depth=frame_depth, frame_weight=frame_weight,
                                                   mask_prob=mask_prob, features=features, 
                                                   window_size_row=window_size_row, window_size_col=window_size_col, 
                                                   weight=weight, return_frame_indices=False)
            x, y_true, mask = batch_data['x'], batch_data['y_true'], batch_data['mask']
            if weight is not None:
                batch_weight = batch_data['weight']
            y_pred = model(x)
            if ndim == 2:
                y_pred = y_pred.squeeze(1)
            elif ndim == 3:
                y_pred = y_pred.squeeze(1).squeeze(1)
            if weight is not None:
                loss = loss_reg_fn((y_pred*batch_weight)[mask], (y_true*batch_weight)[mask])
            else:
                loss = loss_reg_fn(y_pred[mask], y_true[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            if verbose and (i==0 or i==num_iters-1 or (i+1)%print_every==0):
                print(f'i={i+1}, loss={loss.item()}')
        end_time = time.time()
        if verbose:
            print(f'Epoch {epoch+1} time: {end_time - start_time}')
            plt.plot(loss_history[-num_iters:], 'o-', markersize=3)
            plt.ylabel('loss')
            plt.xlabel(f'iteration (epoch {epoch+1})')
            plt.show()
        if save_intermediate_results:
            torch.save(model.state_dict(), f'{save_folder}/model_step{len(loss_history)}.pt')
            np.save(f'{save_folder}/loss__denoise.npy', loss_history)
            torch.cuda.empty_cache()
            denoised_mat = model_denoise(mat[movie_start_idx-frame_depth:movie_end_idx+frame_depth], model, ndim=ndim, frame_depth=frame_depth,
                                             features=features, batch_size=batch_size_eval, normalize=False, replicate_pad=False)
            movie_tyx = denoised_mat * std_mat + mean_mat
            make_video_ffmpeg(movie_tyx, 
                              save_path=f'{save_folder}/denoised_movie_frame{movie_start_idx}to{movie_end_idx}_step{len(loss_history)}.avi', fps=fps)
#             if epoch == num_epochs-1:
            np.save(f'{save_folder}/denoised_movie_frame{movie_start_idx}to{movie_end_idx}_step{len(loss_history)}.npy', 
                    movie_tyx.cpu().numpy())
            del denoised_mat, movie_tyx
        torch.cuda.empty_cache()
        if (loss_threshold > 0 and epoch > 0 and 
            np.abs(np.array(loss_history[-num_iters:]).mean() - np.array(loss_history[-2*num_iters:-num_iters]).mean()) < loss_threshold):
            break
    torch.save(model.state_dict(), f'{save_folder}/model_step{len(loss_history)}.pt')
    np.save(f'{save_folder}/loss__denoise.npy', loss_history)
    if verbose:
        plt.plot(loss_history, 'o-', markersize=3)
        plt.ylabel('loss')
        plt.xlabel(f'iteration')
        plt.title(f'loss_history[{len(loss_history)}] = ({loss_history[-1]:.3e})')
        plt.show()
    denoised_mat = model_denoise(mat, model, ndim=ndim, frame_depth=frame_depth, features=features, batch_size=batch_size_eval, normalize=False,
                                replicate_pad=True)
    denoised_mat = denoised_mat * std_mat + mean_mat
    np.save(f'{save_folder}/denoised_movie_frame0to{mat.shape[0]}_step{len(loss_history)}.npy', denoised_mat.cpu().numpy())
    end_time = time.time()
    print(f'Total time spent: {end_time - initial_start_time}')
    if return_model:
        return denoised_mat, model
    else:
        return denoised_mat