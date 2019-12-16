import numpy as np
import matplotlib.pyplot as plt
import scipy
import warnings

import torch
import torch.nn as nn

from utility import median_absolute_deviation
from visualization import plot_image


def estimate_variance(x, fs=500, plot=False, average='median'):
    """Use Welch's method to estimate noise level (standard deviation)
    
    Parameters:
    -----------
    x: array-like
    fs: sampling frequency
        if fs > x.shape[-1], internally set fs = x.shape[-1]
    
    Examples:
    ---------
        x = np.sin(2 * np.pi * np.linspace(1, 100, 100000).reshape(10, 100, 100))
        x += np.random.randn(*x.shape) * 19
        plt.hist(estimate_variance(x, fs=x.shape[-1], plot=False, average='median').reshape(-1))
        
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    fs = min(fs, x.shape[-1])
    f, Pxx_den = scipy.signal.welch(x, fs=fs, nperseg=fs, average=average)
    if plot:
        plt.semilogy(f, Pxx_den)
    return np.sqrt(Pxx_den[...,fs//4:].mean(-1) * fs / 2)


def get_splits(T, M, D=None, L=None, fixed_L=False): # not used
    """Get data splits (slices) used in Welch's method to calculate Power Spectral Density (PSD) 
    https://en.wikipedia.org/wiki/Welch%27s_method
    
    Args:
        T: the length of the time series
        M: the length of each segment
        D: the length of overlapping between any two adjacent segments
        L: the number of segments; ignored when D is given, otherwise use 
    """
    if D is None:
        # assert L is not None
        stride = (T - M + L) // L # (T - M + 1 + L - 1) // L
        D = M - stride
    stride = M - D
    splits = []
    if L is None or (not fixed_L):
        L = (T - stride + 1) // stride
    for i in range(L):
        if stride*i + M < T:
            splits.append(range(stride*i, stride*i + M))
        else:
            # use circular padding to make sure each segment have the same length M
            splits.append(list(itertools.chain(range(stride*i, T), range(stride*i + M - T))))
            if not fixed_L:
                break
    return splits


def periodogram(signal, window_fn=torch.hann_window, is_train=False): # not used
    if window_fn is not None:
        signal = signal * torch.hamming_window(window_length=signal.size(-1), periodic=False, device=signal.device)
    if not is_train:
        with torch.no_grad():
            dft = torch.rfft(signal, signal_ndim=1, onesided=True)
    return torch.pow(dft, 2).sum(-1)


def power_spectral_density(signal, M=None, D=None, L=None, return_noise_level=False): # not used
    T = signal.size(-1)
    device = signal.device
    if M is None:
        M = T//2
    if D is None:
        D = M//2
    splits = get_splits(T, M, D=D, L=L)
    psd = torch.stack([periodogram(signal.index_select(dim=-1, index=torch.tensor(split, device=device))) 
                       for split in splits], dim=0).mean(0)
    if return_noise_level:
        return torch.sqrt(psd[psd.shape[-1]//2:].mean() / (psd.shape[-1]))

    
def second_order_difference(x):
    # x[t-1] - 2x[t] + x[t+1]
    return (x[:-2] - 2*x[1:-1] + x[2:]).abs().sum()


def total_variation(x, size=None, sel_dim=None):
    """Calculate total variation
    
    Args:
        size: sequence-like or None
            if None, set size = x.size()
    """
    if size is None:
        size = x.size()
    else:
        x = x.reshape(size)
    ndim = x.dim()
    tv = 0
    if sel_dim is None:
        sel_dim = range(ndim)
    for d in sel_dim:
        tv = tv + (x[[slice(1, size[i]) if i==d else slice(None) for i in range(ndim)]] - 
                   x[[slice(size[i]-1) if i==d else slice(None) for i in range(ndim)]]).abs().sum()
    return tv
    
    
def denoise(x, loss_fn=total_variation, max_num_iter=1000, auto_stop=True, verbose=False, true_x=None, 
            lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, return_detached=True, cmap=None):
    """Use total variation or trend filtering penalities as loss to denoise
    """
    threshold = np.sqrt(x.numel()) * estimate_variance(x.reshape(-1))
    loss_history = []
    distance_history = []
    u = x.clone().detach().requires_grad_(True)
    optimizer = torch.optim.AdamW([u], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=False)
    for i in range(max_num_iter):
        def closure():
            optimizer.zero_grad()
            loss = loss_fn(u)
            loss.backward()
            return loss
        optimizer.step(closure)
        if verbose:
            loss_history.append(loss_fn(u).item())
            distance_history.append(torch.norm(u-x).item())
            if i % 100 == 0:
                if i == 0:
                    if true_x is not None:
                        plot_image(true_x, title='True x', cmap=cmap)
                    plot_image(x, title='x', cmap=cmap)
                plot_image(u, title=f'Denoised x at iter={i}', cmap=cmap)
        if auto_stop and torch.norm(u-x) > threshold:
            if verbose:
                print(f'Stop at iteration {i}, distance={torch.norm(u-x).item():.2f} >= threshold={threshold:.2f}')
                plot_image(u, title=f'Denoised x at iter={i}', cmap=cmap)
            break
    if verbose:
        plt.plot(loss_history)
        plt.title('loss')
        plt.show()
        plt.plot(distance_history)
        plt.title(f'|x-u|=distance={distance_history[-1]:.2f}\nthreshold={threshold:.2f}')
        plt.show()
    if return_detached:
        u = u.detach().requires_grad_(False)
    if verbose:
        return u, loss_history, distance_history
    else:
        return u


def spatial_update(R, v, size, return_detached=True):
    u = torch.matmul(R, v).detach()
    u = denoise(u.reshape(size), loss_fn=total_variation, max_num_iter=1000, auto_stop=True, verbose=False, true_x=None, 
            lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, return_detached=return_detached).reshape(-1)
    u = u / torch.norm(u)
    return u


def temporal_update(R, u, return_detached=True):
    v = torch.matmul(R.t(), u)
    v = denoise(v, loss_fn=second_order_difference, max_num_iter=1000, auto_stop=True, verbose=False, true_x=None, 
            lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, return_detached=return_detached).reshape(-1)
    v = v / torch.norm(v)
    return v


# Penalized matrix decomposition for denoising and compressing  
def decimated_initialization(R, scale_factor=[0.5, 0.5], tol=1e-4, mode='nearest', align_corners=None, max_iter_num=100):
    """First down sampling, then upsampling
    
    Args:
        R: torch.Tensor with shape (d, T)
        
    Returns:
        u: torch.Tensor with shape (d, ) and norm=1
    """
    d = R.size(0)
    # When calling squeeze() I did not consider corner cases
    Rds = nn.functional.interpolate(R[None, None], size=None, scale_factor=scale_factor, mode=mode, align_corners=align_corners).squeeze()
    u0 = Rds.new_ones(Rds.size(0))
    u0 = u0 / torch.norm(u0)
    v0 = torch.matmul(Rds.t(), u0)
    v0 = v0 / torch.norm(v0)
    du = dv = float('Inf')
    for _ in range(max_iter_num):
        u = torch.matmul(Rds, v0)
        u = u / torch.norm(u)
        du = torch.norm(u - u0)
        u0 = u
        v = torch.matmul(Rds.t(), u0)
        v = v / torch.norm(v)
        dv = torch.norm(v - v0)
        v0 = v
        if max(du, dv) < tol:
            break
    u = nn.functional.interpolate(u[None, None], size=d, scale_factor=None, mode=mode, align_corners=align_corners).squeeze()
    u = u / torch.norm(u)
    return u


def rank_one_decomposition(R, size, tol=1e-1, max_num_iter=20):
    """Single factor penalized matrix decomposition with total variation and trend filtering penalties
    
    Args:
        R: 2-d tensor
            shape=(d, T), where d = nrow * ncol
        size: (nrow, ncol)
            For calculating total variation
            
    Returns:
        u: torch.Tensor with shape=(d,) and norm=1
        v: torch.Tensor with shape=(T,) and norm=1
        
    """
    u0 = decimated_initialization(R)
    v0 = temporal_update(R, u0)
    du = dv = float('Inf')
    for i in range(max_num_iter):
        u = spatial_update(R, v0, size)
        du = torch.norm(u - u0)
        v = temporal_update(R, u)
        dv = torch.norm(v - v0)
        u0 = u
        v0 = v
        if max(du, dv) <= tol:
            break
    if min(du, dv) > tol:
        msg = f'Reached max_num_iter={max_num_iter}, but not converged!\ndu={du.item():.2e}, dv={dv.item():.2e}, tol={tol:.2e}'
        warnings.warn(msg)
    return u, v


def pmd_compress(R, size, max_num_fails=5, max_num_components=40, tol=1e-1, spatial_threshold=5e-1, temporal_threshold=5e-1, verbose=False):
    """Penalized matrix decomposition for denoising and compressing with total variation and trend filtering penalties
    
    Args:
        R: torch.Tensor with shape=(d, T), where d = nrow * ncol
        size: (nrow, ncol) for calculating total variation penalty
        
    Returns:
        U: torch.Tensor with shape=(d, k) where k in determined during the executation
        V: torch.Tensor with shape=(k, T)
    """
    U = []
    V = []
    num_fails = 0
    while num_fails < max_num_fails and len(U) < max_num_components:
        u, v = rank_one_decomposition(R, size, tol=tol)
        v = torch.matmul(R.t(), u)
        tv = total_variation(u, size=size) / u.abs().sum()
        tf = second_order_difference(v) / v.abs().sum()
        if tv < spatial_threshold and tf < temporal_threshold:
            U.append(u)
            V.append(v)
            num_fails = 0
        else:
            num_fails += 1
        R = R - u.unsqueeze(1) * v
        if verbose:
            print(f'tv={tv.item():.2f}, tf={tf.item():.2f}, num_fails={num_fails}, len(U)={len(U)}')
    U = torch.stack(U, dim=1)
    V = torch.stack(V, dim=0)
    torch.cuda.empty_cache()
    return U, V

def get_threshold(size, loss_fn=total_variation, repeat=100, percentile=1, sampling_fn=torch.randn, plot=False, 
                  device=torch.device('cuda')):
    result = []
    for _ in range(repeat):
        u = sampling_fn(size, device=device)
        loss = loss_fn(u) / u.abs().sum()
        result.append(loss.item())
    result = np.array(result)
    threshold = np.percentile(result, percentile)
    if plot:
        plt.hist(result)
        plt.axvline(threshold, c='r')
        plt.show()
    return threshold

def penalized_matrix_decomposition(Y, size, delta=2, max_num_fails=10, verbose=False):
    """Pipeline for penalized matrix decomposition
    
    Args:
        Y: torch.Tensor with shape=(d, T), where d=nrow*ncol
        size: (nrow, ncol)
    """
    std = estimate_variance(Y, fs=Y.shape[-1]) # numpy dtype
    mean = Y.mean(-1, keepdim=True)
    Y = (Y - mean) / Y.new_tensor(std).unsqueeze(-1)
    U, V = pmd_compress(Y, size, max_num_fails=max_num_fails, verbose=verbose)
    Y = torch.mm(U, V)
    mad, median = median_absolute_deviation(Y, dim=-1, scale=1.)
    Y = nn.functional.relu(Y-median-delta*mad)
    torch.cuda.empty_cache()
    return Y