import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from models import DenseNet
from visualization import imshow


def svd(mat, plot=False, figsize=(15, 10)):
    u, s, v = torch.svd(mat)
    num_pcs = detect_outliers(s, return_outliers=True, filter_fn=lambda points, min_val, max_val: points > max_val).sum().item()
    if num_pcs > 10:
        num_pcs = (s[0] - s[1:-1] < s[1:-1] - s[-1]).sum().item() + 1
    if plot:
        plt.figure(figsize=figsize)
        plt.plot(s.detach().cpu(), 'o--')
        plt.title(f'Significant singular values {num_pcs}')
        plt.show()
    return num_pcs, u, s, v
   
    
def watershed_segment(image, markers=None, plot=False):
    from scipy import ndimage as ndi
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    distance = ndi.distance_transform_edt(image)
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                labels=image)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers=markers, mask=image)
    if plot:
        fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].set_title('Overlapping objects')
        ax[1].imshow(-distance, cmap=plt.cm.gray)
        ax[1].set_title('Distances')
        ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
        ax[2].set_title('Separated objects')
        for a in ax:
            a.set_axis_off()
        fig.tight_layout()
        plt.show()
    return labels
    
    
def read_tiff_file(filepath):
    from skimage import io
    return io.imread(filepath)
#     from PIL import Image, ImageSequence
#     im = Image.open(filepath)
#     return np.array([np.array(page) for page in ImageSequence.Iterator(im)])


def detect_outliers(points, whis=1.5, return_outliers=False, filter_fn=lambda points, min_val, max_val: np.logical_or(points<min_val, points>max_val)):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    q1 = np.percentile(points, 25)
    q3 = np.percentile(points, 75)
    iqr = q3 - q1
    min_val = q1 - whis * iqr
    max_val = q3 + whis * iqr
    mask = filter_fn(points, min_val, max_val)
    if return_outliers:
        return mask
    else:
        return np.logical_not(mask)
    

def empty_cache(cond, delete=True, verbose=True):
    for k in [k for k, v in globals().items() if cond(k, v)]:
        if verbose:
            print(k)
        if delete:
            del globals()[k]
    torch.cuda.empty_cache()

    
def median_absolute_deviation(x, dim=-1, scale=1.):
    """
    Args:
        x: n-d torch.Tensor
    
    Returns:
        mad: median absolute deviation along dim of x, same shape with x
        median: median along dim of x, same shape with x
    """
    median = x.median(dim=dim)[0].unsqueeze(dim)
    mad = (x - median).abs().median(dim=dim)[0].unsqueeze(dim) * scale
    return mad, median

    
def KL_divergence(predict, target):
    return (target * (target / predict).log()).sum() - target.sum() + predict.sum()

def Euclidean(predict, target):
    return ((predict - target)**2).sum()

def IS_divergence(predict, target):
    div = target / predict
    return div.sum() - div.log().sum() - div.numel()

def Beta_divergence(predict, target, beta=2, square_root=True):
    if beta == 2:
        beta_div = Euclidean(predict, target)
    elif beta == 1:
        beta_div = KL_divergence(predict, target)
    elif beta == 0:
        beta_div = IS_divergence(predict, target)
    else:
        beta_div = (target.pow(beta).sum() + (beta-1) * predict.pow(beta).sum() - beta * (target * predict.pow(beta-1)).sum()) / (beta * (beta-1))

        
def weighted_mse_loss(pred, target, weight=None, loss_fn=nn.functional.mse_loss):
    if weight is None:
        return loss_fn(pred, target)
    else:
        return loss_fn(pred*weight, target*weight)
    

def densenet_regression(x, y, hidden_dims, loss_fn=nn.MSELoss(), lr=1e-2, weight_decay=1e-4, num_iters=1000, print_every=100,
                        device=torch.device('cuda'), verbose=True, plot=True):
    in_dim = x.size(-1)
    out_dim = y.size(-1)
    hidden_dims = hidden_dims + [out_dim]
    model = DenseNet(in_dim, hidden_dims, dense=True, residual=False).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay, amsgrad=True)
    for i in range(num_iters):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose and (i % print_every == 0):
            print(i, loss.item())
    if plot:
        plt.plot(x.detach().cpu().numpy().reshape(-1), y.detach().cpu().numpy().reshape(-1), 'ro', 
         x.detach().cpu().numpy().reshape(-1), y_pred.detach().cpu().numpy().reshape(-1), 'g--')
        plt.show()
    return model


def power_series(X, order=1):
    if X.dim() == 1:
        X = X.unsqueeze(1)
    X_expanded = torch.cat([torch.pow(X, i) for i in range(1, order+1)] + [X.new_ones(X.size())], dim=1)
    del X
    torch.cuda.empty_cache()
    return X_expanded


def linear_regression(X, Y=None, order=None, X_test=None):
    if X.dim() == 1:
        X = X.unsqueeze(1)
    if Y is None:
        if order is None:
            beta_left = torch.matmul(torch.matmul(X.t(), X).inverse(), X.t())
            return beta_left
        else:
            assert isinstance(order, int) and order > 0
            X = torch.cat([torch.pow(X, i) for i in range(1, order+1)] + [X.new_ones(X.size())], dim=1)
            beta_left = torch.matmul(torch.matmul(X.t(), X).inverse(), X.t())
            return beta_left
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)
    if order is None:
        beta = torch.matmul(torch.matmul(torch.matmul(X.t(), X).inverse(), X.t()), Y)
        if X_test is None:
            Y_test = torch.matmul(X, beta)
        else:
            Y_test = torch.matmul(X_test, beta)
        return beta, Y_test
    else:
        assert isinstance(order, int) and order > 0
        X = torch.cat([torch.pow(X, i) for i in range(1, order+1)] + [X.new_ones(X.size())], dim=1)
        if X_test is None:
            X_test = X
        else:
            if X_test.dim() == 1:
                X_test = X_test.unsqueeze(1)
            X_test = torch.cat([torch.pow(X_test, i) for i in range(1, order+1)] + [X_test.new_ones(X_test.size())], dim=1)
        return linear_regression(X, Y, order=None, X_test=X_test)

    
def cosine_similarity(a, b, dim=0):
    with torch.no_grad():
        a = a - a.mean(dim, keepdim=True)
        b = b - b.mean(dim, keepdim=True)
        cor = nn.functional.cosine_similarity(a, b, dim=dim)
    return cor

def get_adj_list(cor, nrow, ncol, t=1):
    """Only used in neighbor_cor
    """
    if isinstance(cor, torch.Tensor):
        cor = cor.detach().cpu().numpy()
    left = np.ravel_multi_index(np.nonzero(cor>0), (nrow, ncol))
    if t == 1: 
        right = left + ncol
    elif t == 2:
        right = left + 1
    elif t == 3:
        right = left + ncol + 1
    elif t == 4:
        right = left + ncol - 1
    weight = cor[cor>0]
    return np.stack([left, right, weight], axis=1)

def neighbor_cor(mat, neighbors=8, choice='mean', nonnegative=True, return_adj_list=False, plot=False, 
                 title='Correlation Map'):
    """Assume mat has shape (D, nrow, ncol)
    """
    cor1 = cosine_similarity(mat[:, 1:], mat[:, :-1], dim=0) # row shift
    cor2 = cosine_similarity(mat[:, :, 1:], mat[:, :, :-1], dim=0) # column shift
    cor3 = cosine_similarity(mat[:, 1:, 1:], mat[:, :-1, :-1], dim=0) # diagonal 135
    cor4 = cosine_similarity(mat[:, 1:, :-1], mat[:, :-1, 1:], dim=0) # diagonal 45
    nrow, ncol = mat.shape[1:]
    if return_adj_list:
        adj_list = np.concatenate([get_adj_list(c, nrow, ncol, i+1) for i, c in enumerate([cor1, cor2, cor3, cor4])], 
                                  axis=0)
        return adj_list
    with torch.no_grad():
        cor = mat.new_zeros(mat.shape[1:])
        if choice == 'mean':
            cor[:-1] += cor1
            cor[1:] += cor1
            cor[:, :-1] += cor2
            cor[:, 1:] += cor2
            if neighbors == 4:
                denominators = [4, 3, 2]
            elif neighbors == 8:
                denominators = [8, 5, 3]
                cor[1:, 1:] += cor3
                cor[:-1, :-1] += cor3
                cor[1:, :-1] += cor4
                cor[:-1, 1:] += cor4
            else:
                raise ValueError(f'neighbors={neighbors} is not implemented!')
            cor[1:-1, 1:-1] /= denominators[0]
            cor[0, 1:-1] /= denominators[1]
            cor[-1, 1:-1] /= denominators[1]
            cor[1:-1, 0] /= denominators[1]
            cor[1:-1, -1] /= denominators[1]
            cor[0, 0] /= denominators[2]
            cor[0, -1] /= denominators[2]
            cor[-1, 0] /= denominators[2]
            cor[-1, -1] /= denominators[2]
        elif choice == 'max':
            cor[:-1] = torch.max(cor[:-1], cor1)
            cor[1:] = torch.max(cor[1:], cor1)
            cor[:, :-1] = torch.max(cor[:, :-1], cor2)
            cor[:, 1:] = torch.max(cor[:, 1:], cor2)
            if neighbors == 8:
                cor[1:, 1:] = torch.max(cor[1:, 1:], cor3)
                cor[:-1, :-1] = torch.max(cor[:-1, :-1], cor3)
                cor[1:, :-1] = torch.max(cor[1:, :-1], cor4)
                cor[:-1, 1:] = torch.max(cor[:-1, 1:], cor4)
        else:
            raise ValueError(f'choice = {choice} is not implemented!')
    if plot:
        imshow(mat.mean(0), title='Temporal Mean')
        imshow(cor, title=title)
    if nonnegative:
        cor = torch.nn.functional.relu(cor, inplace=True)
    for k in [k for k in locals().keys() if k!='cor']:
        del locals()[k]
    torch.cuda.empty_cache()
    return cor


def get_label_image(image, min_pixels=50, square_width=3, thresh=None,
                    connectivity=None, plot=False, figsize=(20, 10)):
    """Use skimage to filter background and get labels
    
    Args:
        image: 2-D numpy array or torch.Tenor
        
    Returns:
        label_image: same size with image
        regions: skimage.measure.regionprops(image)
    """
    from skimage.measure import label, regionprops
    from skimage.morphology import closing, square
    from skimage.filters import threshold_otsu
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if thresh is None:
        # apply threshold
        thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(width=square_width))
    if plot:
        plt.figure(figsize=figsize)
        plt.imshow(bw)
        plt.title(f'threshold={thresh:.2f}')
        plt.show()
    # label image regions
    label_image = label(bw, connectivity=connectivity)
    for k in np.unique(label_image):
        loc = label_image==k
        if loc.sum() < min_pixels:
            label_image[loc] = 0
    for i, k in enumerate(sorted(np.unique(label_image))):
        if i < k:
            label_image[label_image==k] = i
    regions = regionprops(label_image)
    return label_image, regions

def split_label(split_label, split_label_segmentation, label_image, regions=None):
    """Assume label_image contains 0 (background) and labels from 1 to label_image.max()
    split_label_segmentation contains 0 as background and positive integers as labels
    """
    if regions is None:
        regions = regionprops(label_image)
    minr, minc, maxr, maxc = regions[split_label-1].bbox
    for i, l in sorted(enumerate(np.unique(split_label_segmentation)))[1:]:
        label_image[minr:maxr, minc:maxc][split_label_segmentation==l] = split_label if i == 0 else label_image.max()+1
    return label_image

def find_max_dict(dic):
    m = float('-Inf')
    for k, v in dic.items():
        if v > m:
            m = v
            key = k
    return key, m

def get_seg(diff, topk=100, min_dist=500, verbose=True): # depleted
    """Get the segmentation
    
    topk: a parameter to identify the boundaries of each segmentation
    
    """
    idx = diff.topk(topk)[1].sort()[0]
    start = []
    end = []
    for i in range(topk-1):
        if idx[i+1] - idx[i] > min_dist:
            start.append(idx[i].item())
            end.append(idx[i+1].item()+1)
    start.append(idx[i+1].item())
    end.append(len(diff)+1)
    if verbose:
        for i, (s, e) in enumerate(zip(start, end)):
            print(i, e-s)
    return start, end


def get_sub_seg(seg_mean, delta=5, shift_left=20, len_left=100, shift_right=30, len_right=100, plot=True): # no longer used
    if isinstance(seg_mean, torch.Tensor):
        seg_mean = seg_mean.detach().cpu().numpy().reshape(-1)
    seg_mean_diff = seg_mean[delta:] - seg_mean[:-delta]
    s = int(np.argmax(seg_mean_diff)) + shift_left // 2
    e = int(np.argmin(seg_mean_diff[s:])) + s
    e_left = s - shift_left
    s_left = e_left - len_left
    s_right = e + shift_right
    e_right = s_right + len_right
    if plot:
        plt.plot(seg_mean)
        plt.axvline(x=s, c='r', ls='--')
        plt.axvline(x=e, c='r', ls='--')
        plt.axvline(x=s_left, c='k', ls='--')
        plt.axvline(x=e_left, c='k', ls='--')
        plt.axvline(x=s_right, c='k', ls='--')
        plt.axvline(x=e_right, c='k', ls='--')
        plt.title('seg_mean')
        plt.show()
        plt.plot(seg_mean_diff)
        plt.title('seg_mean_diff')
        plt.show()
    return (s, e+1), (s_left, e_left+1), (s_right, e_right+1)
    

def get_seg_adj(seg, fixed=False, s=0, e=0, s_left=0, e_left=0, s_right=0, e_right=0, train_size_left=50, train_size_right=50, 
                shift_left=10, shift_right=100, linear_reg_order=None, device=torch.device('cuda'), plot=True, verbose=True): # no longer used
    seg_mean = seg.reshape(len(seg), -1).mean(1)
    if fixed:
        if e==0:
            e = len(seg)
        if e_left==0:
            e_left = s_left + train_size_left
        if e_right==0:
            e_right = len(seg)
        if s_right==0:
            s_right = e_right - train_size_right
    else:
        (s, e), (s_left, e_left), (s_right, e_right) = get_sub_seg(seg_mean, shift_left=shift_left, 
                                                                   shift_right=shift_right, plot=plot)
    if s_left < 0 or e_right > len(seg):
        print(f'The segments are beyond boundaries. Check')
        return seg[s:e], None, (s, e)
    
    input_aug = torch.linspace(-2, 2, e_right-s_left).to(device)
    x_train = torch.cat([input_aug[:e_left-s_left], input_aug[s_right-s_left:e_right-s_left]])
    y_train = torch.cat([seg_mean[s_left:e_left], seg_mean[s_right:e_right]])
    y_train_mean = y_train.mean()
    y_train_std = y_train.std()
    y_train = (y_train - y_train_mean) / y_train_std
    x_train = x_train.unsqueeze(-1)
    y_train = y_train.unsqueeze(-1)
    if linear_reg_order is not None:
        beta, y_pred_linear = linear_regression(x_train, y_train, order=linear_reg_order, X_test=input_aug.unsqueeze(-1)) 
        y_pred_linear = y_pred_linear * y_train_std + y_train_mean

    hidden_dims = [100, 100]
    loss_fn = nn.MSELoss()
    lr = 1e-2
    weight_decay = 1e-4
    num_iters = 1000
    print_every = 100
    model = densenet_regression(x_train, y_train, hidden_dims, loss_fn=loss_fn, lr=lr, 
                                weight_decay=weight_decay, num_iters=num_iters, print_every=print_every, 
                                device=device, verbose=verbose, plot=plot)
    y_pred = model(input_aug.unsqueeze(-1))*y_train_std + y_train_mean
    seg_adj = seg[s:e] - y_pred[s-s_left:e-s_left].unsqueeze(-1)
    if plot:
        plot_tensor(x_train, y_train*y_train_std + y_train_mean, 'ro')
        plot_tensor(input_aug, seg_mean[s_left:e_right], 'y+')
        plot_tensor(input_aug, y_pred, 'g--')
        if linear_reg_order is not None:
            plot_tensor(input_aug, y_pred_linear, 'k-.')
        plt.show()
        plot_tensor(input_aug, seg_mean[s_left:e_right] - y_pred.squeeze(), 'g--')
        plt.title('Detrended segment')
        plt.show()
    return seg[s:e], seg_adj.detach(), (s, e)
