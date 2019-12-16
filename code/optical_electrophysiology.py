import os, functools

import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.cluster import KMeans

import torch
import torch.nn as nn

from utility import linear_regression, power_series, empty_cache, get_label_image, neighbor_cor, svd
from models import get_bg_mat
from visualization import plot_tensor, plot_image_label_overlay, imshow, plot_hist, plot_curves, plot_singular_values
from train import step_decompose


def refine_segmentation(submat, label_mask, label_image, label_idx, plot=False, figsize=(15, 10)):
    num_pcs, u, s, v = svd(submat[:, label_mask.bool()], plot=plot)
    if num_pcs >= 2:
        A, B, loss_history = step_decompose(submat.reshape(submat.size(0), -1), num_components=2*num_pcs)
        mask = B[0].reshape(submat.size(1), submat.size(2))
        X = mask.reshape(-1).unsqueeze(-1)
        kmeans = KMeans(n_clusters=2*num_pcs, random_state=0).fit(X.detach().cpu())
        split_label_segmentation = kmeans.labels_.reshape(mask.shape)
        label_image, regions = get_label_image(image=None, label_image=label_image, split_label=label_idx-1,
                                               split_label_segmentation=split_label_segmentation, plot=plot, figsize=figsize)


def get_size_from_txt(filepath):
    meta_data = pandas.read_csv(filepath, sep='\t', header=None, index_col=0)
    size = int(meta_data.loc['frames']), int(meta_data.loc['ywidth']), int(meta_data.loc['xwidth'])
    return size

def load_file(filepath, size=-1, dtype=np.uint16, device=torch.device('cuda')):
    array = np.fromfile(filepath, dtype=np.uint16).reshape(size).astype('float32')
    mat = torch.from_numpy(array).to(device)
    return mat

def load_mat(exp_id, meta_data, folder, device=torch.device('cuda')):
    file_meta = meta_data[exp_id]
    ncol, nrow, L = file_meta['xwidth'], file_meta['ywidth'], file_meta['frames']
    mat = load_file(filepath=os.path.join(folder, exp_id + '.bin'), size=[L, nrow, ncol], dtype=np.uint16, device=device)
    return mat

def plot_mean_intensity(mat, detrended=None, plot_detrended=False, plot_segments=False, num_frames=3000, period=500, signal_length=100, 
                        figsize=(20, 10)):
    array = mat.mean(-1).mean(-1).cpu()
    if detrended is not None and plot_detrended:
        array = array - array.min()
    xs = [array]
    colors = ['b']
    labels = ['mean intensity']
    if detrended is not None:
        detrended = detrended.mean(-1).mean(-1).cpu()
        if plot_detrended:
            detrended = detrended - detrended.min()
        xs.append(array - detrended)
        colors.append('k')
        labels.append('trend')
        if plot_detrended:
            xs.append(detrended)
            colors.append('g')
            labels.append('detrended')
    plot_curves(xs, colors, labels, show=False, marker='o', linestyle='--', linewidth=1, markersize=1)
    for i in range(0, num_frames, period):
        plt.axvline(x=i, color='r', linestyle='-.')
        plt.axvline(x=i+signal_length, color='r', linestyle='-.')
        plt.axvline(x=i, color='g', linestyle='--')
        plt.axvline(x=i+period, color='g', linestyle='--')
    plt.show()
    if plot_segments:
        for i in range(0, num_frames, period):
            plot_curves([x[i:i+period] for x in xs], colors, labels, show=False, marker='o', linestyle='--', linewidth=1, markersize=1)
            plt.axvline(x=0, color='r', linestyle='-.')
            plt.axvline(x=signal_length, color='r', linestyle='-.')
            plt.title(f'segment {i//period}')
            plt.show()

def detrend_linear(mat, train_idx=None, linear_order=3, input_min=-2, input_max=2, return_trend=False,
                   device=torch.device('cuda')):
    input_aug = torch.linspace(input_min, input_max, mat.shape[0], device=device)
    if train_idx is None:
        train_idx = range(mat.shape[0])
    beta, trend = linear_regression(X=input_aug[train_idx], Y=mat.reshape(mat.shape[0], -1)[train_idx], order=linear_order,
                                    X_test=input_aug)
    trend = trend.reshape(mat.shape)
    mat_adj = mat - trend
    if return_trend:
        for k in [k for k in locals().keys() if k not in ['mat_adj', 'trend']]:
            del locals()[k]
        torch.cuda.empty_cache()
        return mat_adj, trend
    else:
        for k in [k for k in locals().keys() if k not in ['mat_adj']]:
            del locals()[k]
        torch.cuda.empty_cache()
        return mat_adj

def detrend(mat, start0, end0, train_size_left, train_size_right, linear_order=3, use_mean_bg=False, plot=False, test_left=None, test_right=None, 
            device=torch.device('cuda'), exp_id=None, meta_data=None, folder=None, show_singular_values=False, **kwargs):
    if mat is None:
        mat = load_mat(exp_id, meta_data, folder) # from **kwargs
    _, nrow, ncol = mat.shape
    mat_list = [mat[s:e] for s, e in zip(start0, end0)]
    length0 = end0[0] - start0[0]
    input_aug = torch.linspace(-2, 2, length0, device=device)
    x_train = torch.cat([input_aug[:train_size_left], input_aug[-train_size_right:]])
    beta_left = linear_regression(x_train, order=linear_order)
    
    mat_adj = []
    if use_mean_bg:
        kernel_size = 7 # odd number (>=3) to maintain the same shape
        bg_mat = get_bg_mat(mat, kernel_size)

    for i in range(len(start0)): 
        # Pixel-level detrending
        y_bg = bg_mat if use_mean_bg else mat
        y_bg = y_bg[start0[i]:end0[i]].reshape(length0, -1)
        y_train = torch.cat([y_bg[:train_size_left], y_bg[-train_size_right:]])
        y_train_mean = y_train.mean()
        y_train_std = y_train.std()
        y_train = (y_train - y_train_mean) / y_train_std
        beta = torch.matmul(beta_left, y_train)
        y_pred = torch.matmul(power_series(input_aug, order=linear_order), beta)
        y_pred = y_pred * y_train_std + y_train_mean
        y_true = mat[start0[i]:end0[i]].reshape(length0, -1)
        y_adj = y_true - y_pred
        mat_adj.append(y_adj.reshape(length0, nrow, ncol))

        if plot:
            if show_singular_values:
                plot_singular_values(y_true, marker='o--', linewidth=1, markersize=2, use_cpu=True, end=20, show=True,
                                     title=f'Segment {i+1} singular values BEFORE detrending')
                plot_singular_values(y_adj, marker='o--', linewidth=1, markersize=2, use_cpu=True, end=20, color='orange',
                                     title=f'Segment {i+1} singular values AFTER detrending')
            plt.scatter(range(length0), y_true.mean(1).cpu(), marker='o', 
                        c=['r']*train_size_left + ['b']*(length0-train_size_left-train_size_right) + ['r']*train_size_right, 
                        s=1, alpha=0.5)
            plot_tensor(y_pred.mean(1), marker='g--', alpha=0.5)
            plt.axvline(test_left, color='g', linestyle='-.')
            plt.axvline(test_right, color='g', linestyle='-.')
            plt.title(f'Segment {i+1}')
            plt.show()
            plot_tensor(y_adj.mean(1), marker='o-', markersize=1, alpha=0.8)
            plt.axvline(test_left, color='g', linestyle='-.')
            plt.axvline(test_right, color='g', linestyle='-.')
            plt.title(f'Detrended segment {i+1}: min={y_adj.min().item():.0f} mean={y_adj.mean().item():.0f} max={y_adj.max().item():.0f}')
            plt.show()
    return mat_adj

def detrend_high_magnification(mat, skip_segments=1, num_segments=6, period=500, train_size_left=0, train_size_right=350, 
                               linear_order=3, plot=False, signal_start=0, signal_end=100, filepath=None, size=(-1, 180, 300), 
                               device=torch.device('cuda'), start0=None, end0=None, return_mat=False, **kwargs):
    if mat is None:
        mat = load_file(filepath=filepath, size=size, dtype=np.uint16, device=device)
    L, nrow, ncol = mat.size()
    if period == 'unknown':
        period = L
    if signal_end == 'period':
        signal_end = period
    train_idx = ([range(skip_segments*period)] +
                 [range(i*period, train_size_left+i*period) for i in range(skip_segments, num_segments)] + 
                 [range((i+1)*period - train_size_right, (i+1)*period) for i in range(skip_segments, num_segments)])
    train_idx = functools.reduce(lambda x,y: list(x)+list(y), train_idx)
    input_aug = torch.linspace(-2, 2, L, device=device)
    beta, trend = linear_regression(X=input_aug[train_idx], Y=mat.reshape(L, -1)[train_idx], order=linear_order, X_test=input_aug)
    mat_adj = mat - trend.reshape(L, nrow, ncol)
    if plot:
        frame_mean = mat.mean(-1).mean(-1)
        plt.figure(figsize=(20, 10))
        plt.plot(frame_mean.cpu(), 'o--', linewidth=1, markersize=2)
        for i in range(num_segments):
            plt.axvline(i*period+signal_start, color='g', linestyle='-.')
            plt.axvline(i*period+signal_end, color='g', linestyle='-.')
        plt.title('Frame mean intensity')
        plt.show()
        imshow(mat.mean(0), title='Mean intensity')
        imshow(trend.mean(0).reshape(nrow, ncol), title='Trend')
        imshow(mat_adj.mean(0).reshape(nrow, ncol), title='Detrended')

        cor = neighbor_cor(mat, neighbors=8, plot=True, choice='max', title='cor mat')
        plot_hist(cor, show=True)
        cor = neighbor_cor(trend.reshape(-1, nrow, ncol), neighbors=8, plot=True, choice='max', title='cor trend')
        plot_hist(cor, show=True)
        cor = neighbor_cor(mat_adj.reshape(-1, nrow, ncol), neighbors=8, plot=True, choice='max', title='cor mat_adj')
        plot_hist(cor, show=True)
    
        plot_mean_intensity(mat, detrended=mat_adj, plot_detrended=True, plot_segments=True, num_frames=L, period=period, 
                            signal_length=signal_end-signal_start)
    if start0 is not None and end0 is not None:
        mat_adj = [mat_adj[s:e] for s, e in zip(start0, end0)]
    if return_mat:
        return mat, mat_adj
    else:
        return mat_adj


def extract_super_pixels(mat_adj=None, test_left=None, test_right=None, mat_cat=None, num_neighbors=8, cor_choice='mean', connectivity=None, 
                         min_pixels=50, image=None, plot=False, use_mean_image=False):
    if image is None:
        if mat_cat is None:
            mat_cat = torch.cat([m[test_left:test_right] for m in mat_adj], dim=0)
        cor_global = neighbor_cor(mat_cat, neighbors=num_neighbors, choice=cor_choice, plot=plot, 
                                  title='correlation map')
        if use_mean_image:
            cor_global = mat_cat.mean(0) * cor_global
            cor_global = cor_global / cor_global.max()
        image = cor_global.detach().cpu().numpy()
    else:
        cor_global = image # for backward compatibility
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    label_image, regions = get_label_image(image, min_pixels=min_pixels, connectivity=connectivity, plot=False)
    if plot:
        plot_image_label_overlay(image, label_image)
    return cor_global, label_image, regions

def get_percentile(a, percentile):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    return np.percentile(a.reshape(-1), q=percentile)
    
def extract_single_trace(mat, label_mask, percentile=50):
    binary_mask = label_mask > 0
    if percentile > 0:
        binary_mask = label_mask > get_percentile(label_mask[binary_mask], percentile)
    trace = (mat*label_mask*binary_mask).sum(-1).sum(-1) / label_mask[binary_mask].sum()
    return trace

def extract_traces(mat, softmask, label_image, regions=None, percentile=50):
    """
    Args:
        label_image: background: 0, labels: 1, 2, 3, ... (no skipping)
    """
    # assert len(np.unique(label_image)) == label_image.max()
    if regions is None:
        regions = regionprops(label_image)
    submats = []
    traces = []
    for i, region in enumerate(regions):
        minr, minc, maxr, maxc = region.bbox
        submat = mat[:, minr:maxr, minc:maxc]
        sub_image = label_image[minr:maxr, minc:maxc]
        label_mask = softmask[minr:maxr, minc:maxc].clone()
        label_mask[torch.from_numpy(sub_image!=i+1)] = 0
        trace = extract_single_trace(submat, label_mask, percentile=percentile)
        submats.append(submat)
        traces.append(trace)
    for k in [k for k in locals().keys() if k not in ['submats', 'traces']]:
        del locals()[k]
    torch.cuda.empty_cache()
    return submats, traces

def get_submat_traces(regions, label_image, seg_idx=0, mat_adj=None, sig_list=None, mat_list=None, mat=None, cor=None,
                      weighted_denominator=True, 
                      weight_percentile=50, return_name='all', linear_order=3, input_aug=None, beta_left=None, train_size_left=None, 
                      train_size_right=None, compare=False, test_left=None, test_right=None, plot_singular_values=False, 
                      device=torch.device('cuda'), **kwargs):
    """Use four different methods to calculate traces
    
    'mat_adj': use pre-calculated detrended matrices with linear regression
    'mean_bg': use the mean background values to detrend
    'y_adj': use the background to detrend with linear regression
    'sig_list': use the original values without detrending
    
    
    """
    def get_trace(mat=None, mat_adj=None, seg_idx=None, submat=None, weight_percentile=weight_percentile, 
                  plot_singular_values=plot_singular_values):
        if submat is None:
            if mat is None:
                mat = mat_adj[seg_idx]
            submat = mat[:, minr:maxr, minc:maxc]
        if cor is None:
            cor_ = neighbor_cor(submat, neighbors=8, plot=False, choice='max', title='cor', return_adj_list=False)
        else:
            cor_ = cor[minr:maxr, minc:maxc]
        if weight_percentile is None:
            weight = cor_
        else:
            weight = nn.functional.threshold(cor_, np.percentile(cor_[label_mask.bool()].cpu(), weight_percentile), 0, inplace=False)
        if weight.sum() == 0:
            weight = 1
        denominator = (label_mask*weight).sum() if weighted_denominator else label_mask.sum()
        trace = ((submat * label_mask * weight).sum(-1).sum(-1) / denominator)
        if plot_singular_values:
            u, s, v = torch.svd(submat.reshape(len(submat), -1))
            plot_tensor(s, marker='o--', linewidth=1, figsize=(20, 10), show=True)
        return submat, trace, weight
        
    is_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    traces = {'mat_adj': [], 'sig_list': [], 'mean_bg': [], 'y_adj': []}
    submats = {'mat_adj': [], 'sig_list': [], 'mean_bg': [], 'y_adj': []}
    if mat is not None:
        traces['mat'] = []
        submats['mat'] = []
    num_labels = len(regions)
    if isinstance(label_image, np.ndarray):
        label_image = torch.from_numpy(label_image).to(device=device)
    nrow, ncol = label_image.shape
    
    for label_idx in range(1, num_labels+1):
        minr, minc, maxr, maxc = regions[label_idx-1].bbox
        label_mask = (label_image==label_idx).float()[minr:maxr, minc:maxc] # there can be other labels
        bg_mask = (label_image==0).float()[minr:maxr, minc:maxc]
        if mat is not None:
            submat, trace, weight = get_trace(mat=mat)
            submats['mat'].append([submat, label_mask, weight])
            traces['mat'].append(trace)
        else:
            if mat_adj is not None and return_name == 'all' or return_name == 'mat_adj':
                if seg_idx is None:
                    # this will consume a lot of gpu memory; never used
                    submat =  torch.cat([m[test_left:test_right] for m in mat_adj], dim=0)[:, minr:maxr, minc:maxc]
                    submat, trace, weight = get_trace(submat=submat)
                else:
                    submat, trace, weight = get_trace(mat_adj=mat_adj, seg_idx=seg_idx)
                submats['mat_adj'].append([submat, label_mask, weight])
                if compare:
                    traces['mat_adj'].append(trace[test_left:test_right])
                else:
                    traces['mat_adj'].append(trace)

            if sig_list is not None and return_name == 'all' or return_name == 'sig_list':
                if seg_idx is None:
                    # this will consume a lot of gpu memory; never used
                    submat =  torch.cat(sig_list, dim=0)[:, minr:maxr, minc:maxc]
                    submat, trace, weight = get_trace(submat=submat)
                else:
                    submat, trace, weight = get_trace(mat_adj=sig_list, seg_idx=seg_idx)
                submats['sig_list'].append([submat, label_mask, weight])
                traces['sig_list'].append(trace)

                if return_name == 'all' or return_name == 'mean_bg':
                    y_bg = (submat * bg_mask).sum(-1).sum(-1) / bg_mask.sum()
                    submat = submat - y_bg[:, None, None]
                    submat, trace, weight = get_trace(submat=submat)
                    submats['mean_bg'].append([submat, label_mask, weight])
                    traces['mean_bg'].append(trace)
            
            if mat_list is not None and return_name == 'all' or return_name == 'y_adj':
                if label_idx == 1:
                    if input_aug is None:
                        input_aug = torch.linspace(-2, 2, len(mat_list[seg_idx]), device=device)
                    if beta_left is None:
                        x_train = torch.cat([input_aug[:train_size_left], input_aug[-train_size_right:]])
                        beta_left = linear_regression(x_train, order=linear_order)
                if seg_idx is None:
                    submat =  torch.cat([m[test_left:test_right] for m in mat_list], dim=0)[:, minr:maxr, minc:maxc]
                else:
                    submat = mat_list[seg_idx][:, minr:maxr, minc:maxc]
                y_true = submat.reshape(len(submat), -1)
                y_bg = (submat * bg_mask).reshape(len(submat), (maxr-minr)*(maxc-minc)).sum(1, keepdim=True) / bg_mask.sum()
                y_train = torch.cat([y_bg[:train_size_left], y_bg[-train_size_right:]])
                y_train_mean = y_train.mean()
                y_train_std = y_train.std()
                y_train = (y_train - y_train_mean) / y_train_std
                beta = torch.matmul(beta_left, y_train)
                y_pred = torch.matmul(power_series(input_aug, order=linear_order), beta)
                y_pred = y_pred * y_train_std + y_train_mean
                y_adj = y_true - y_pred
                submat = y_adj.reshape(len(y_adj), maxr-minr, maxc-minc)
                submat, trace, weight = get_trace(submat=submat)
                submats['y_adj'].append([submat, label_mask, weight])
                if compare:
                    traces['y_adj'].append(trace[test_left:test_right])
                else:
                    traces['y_adj'].append(trace)
    torch.cuda.empty_cache()
    torch.set_grad_enabled(is_grad_enabled)
    if return_name == 'all':
        return traces, submats
    else:
        return traces[return_name], submats[return_name]

    
def extract_one_label_data(submats, label_idx):
    submat, label_mask, weight = submats[label_idx]
    X = torch.log1p(submat - submat.min())
    x = X.unsqueeze(0).unsqueeze(0)
    y = x * label_mask
    return x, y, label_mask, weight


def prep_train_data(seg_idx, label_idx, label_image, regions, sig_list=None, mat_list=None, mat_adj=None, cor=None, 
                    return_name='mat_adj'):
    traces, submats = get_submat_traces(seg_idx=seg_idx, regions=regions, label_image=label_image, mat_adj=mat_adj, sig_list=sig_list, 
                                        mat_list=mat_list, cor=cor, weighted_denominator=True, return_name=return_name, compare=False)
    x, y, label_mask, weight = extract_one_label_data(submats, label_idx=label_idx)
    trace = traces[label_idx]
    return x, y, trace, label_mask, weight

