import re, math, collections, itertools

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

import torch
import torch.nn as nn

from models import DenseNet
from visualization import imshow

def scale_and_shift(mat, scale=1., shift=0):
    return (mat - mat.min()) / (mat.max() - mat.min()) * scale + shift
    
def simple_linear_regression(x, y, return_fitted=False):
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    a = ((x*y).sum() - n*x_mean*y_mean) / ((x*x).sum() - n*x_mean*x_mean)
    b = y_mean - a*x_mean
    if return_fitted:
        return a*x + b
    return a, b

def mark_points_in_intervals(points, intervals):
    """
    Args:
        points: (n_points, n_dims)
        intervals: (n_intervals, n_dims, 2); the last dimension [min, max]
    
    Returns:
        selected: bool (n_points, n_intervals)
    """
    n_points, n_dims = points.shape
    for dim in range(n_dims):
        mask = (points[:, dim].unsqueeze(1) >= intervals[:, dim, 0]) & (points[:, dim].unsqueeze(1) <= intervals[:, dim, 1])
        if dim == 0:
            selected = mask
        else:
            selected = mask & selected
    return selected

def adaptive_avg_pool(mat, size):
    ndim = len(size)
    model = getattr(nn, f'AdaptiveAvgPool{ndim}d')(size).to(mat.device)
    with torch.no_grad():
        mat_pooled = model(mat.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return mat_pooled

def get_prime_factors(n):
    prime_factors = []
    while n % 2 == 0: 
        prime_factors.append(2) 
        n = n / 2
    for i in range(3, int(math.sqrt(n))+1,2):
        while n % i== 0: 
            prime_factors.append(i)
            n = n / i 
    if n > 2: 
        prime_factors.append(int(n))
    return prime_factors

def select_names(pattern, names):
    return [n for n in names if re.search(pattern, n)]

def query_complex_dict(dictionary, preferred_key, key, secondary_key_list=[]):
    """dictionary is a dictionary of dictionaries with multiple level hierarchies (often related to json file)
    preferred_key and key are the first-level key
    When preferred_key is available, use it instead of key
    Assume at least one of them can satisfy the query of key_list
    
    Examples:
        d = {'global': {'level1': {'level2': {'level3': 999}}}, 'file1': {}}
        assert query_complex_dict(d, preferred_key='global', key='file1', secondary_key_list=['level1', 'level2', 'level3']) == 999
    """
    if preferred_key in dictionary:
        ans = dictionary[preferred_key]
        for k in secondary_key_list:
            if k in ans:
                ans = ans[k] # ans could end up to be {}, [], '', or None
            else:
                ans = None
                break
    else:
        ans = None
    if ans is None or ans=={} or ans=='' or ans==[]:
        ans = dictionary[key]
        for k in secondary_key_list:
            ans = ans[k]
    return ans

def extract_int_from_string(pattern, string):
    """
    Warnings:
        re.search only finds the first match!
        
    Example:
        pattern = 'FOV'
        string = 'abcd.FOV002.bin'
        assert extract_int_from_string(pattern, string) == 2
    """
    match = re.search(pattern, string)
    if match:
        start = match.span()[-1]
        for i in range(start, len(string)):
            try:
                int(string[i])
            except ValueError:
                break
        num = int(string[start:i])
        return num
    
def try_convert_str_to_num(v, strip_str='""[](),:\n '):
    v = v.strip(strip_str)
    if v.lower() == 'true':
        return True
    if v.lower() == 'false':
        return False
    try: 
        v = int(v)
    except ValueError:
        try:
            v = float(v)
        except ValueError:
            pass
    return v 

def parse_seq(seq):
    """Parse a list of strings
    
    Examples:
        seq = 'row_name 1 (2, 3) true [4.1, 5.0]'.split()
        assert parse_seq(seq) == ['row_name', 1, (2, 3), True, (4.1, 5.0)]
    """
    i = 0
    processed_seq = []
    while i < len(seq):
        s = seq[i]
        if s.startswith('(') or s.startswith('['):
            start_idx = i
            while not seq[i].endswith(')') and not seq[i].endswith(']'):
                i += 1
            tmp = tuple(try_convert_str_to_num(v) for v in ' '.join(seq[start_idx:i+1]).split())
            tmp = tuple(e for e in tmp if e!='')
        elif re.search('[0-9]x', s): # handles use cases by Trinh such as '256x 256'
            tmp = (int(s[:-1]), int(seq[i+1]))
            i += 1
        else:
            tmp = try_convert_str_to_num(s)
        processed_seq.append(tmp)
        i += 1
    return processed_seq

def get_topk_indices(tensor, k, return_values=False):
    """
    Examples:
        tmp = torch.randn(2, 4, 5, 7, 8)
        values, positions = get_topk_indices(tmp, tmp.numel(), return_values=True)
        assert torch.norm(tmp[tuple(positions[:, i] for i in range(positions.shape[1]))] - values) == 0
    """
    values, indices = tensor.view(-1).topk(k=k)
    size = np.cumprod(list(reversed(tensor.shape)))
    if len(size) == 1:
        positions = indices.detach().cpu().numpy()
    else:
        positions = []
        for idx in indices:
            idx = idx.item()
            pos = [idx // size[-2]]
            for i in reversed(range(1, len(size)-1)):
                idx = idx - pos[-1] * size[i]
                pos.append(idx // size[i-1])
            pos.append(idx - pos[-1] * size[0])
            positions.append(pos)
        positions = np.array(positions)
    if return_values:
        return values, positions
    else:
        return positions

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


def get_label_image(image=None, min_pixels=50, square_width=3, thresh=None, connectivity=None, semantic_segmentation=None, min_thresh=0.1, 
                    label_size_max_portion=0.5, plot=False, figsize=(20, 10)):
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
    if semantic_segmentation is None:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if thresh is None:
            # apply threshold
            thresh = threshold_otsu(image)
        if min_thresh is not None:
            thresh = max(thresh, min_thresh)
        semantic_segmentation = closing(image > thresh, square(width=square_width))
    if plot:
        plt.figure(figsize=figsize)
        plt.imshow(semantic_segmentation)
        plt.title(f'Semantic segmentation with threshold={thresh:.2f}')
        plt.show()
    # label image regions
    label_image = label(semantic_segmentation, connectivity=connectivity)
    if plot:
        plt.figure(figsize=figsize)
        plt.imshow(label_image)
        plt.title(f'label image ({collections.Counter(label_image.reshape(-1))} before post-processing)')
        plt.show()
    for k in np.unique(label_image):
        loc = label_image==k
        if loc.sum() < min_pixels or loc.sum() > label_image.size*label_size_max_portion:
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
    
def get_slice_indices(shape, dim):
    pos = [torch.tensor(range(shape[i])) for i in range(dim)]
    for i, p in enumerate(pos):
        for j in range(dim-i):
            p.unsqueeze_(-1)
    return pos

def reset_boundary(mat, b=1, value=0, dim=None):
    mat = mat.float().clone()
    if isinstance(b, int):
        b = [[b, b]] * mat.ndim
    dims = range(mat.ndim) if dim is None else [dim]
    for i in dims:
        mat[get_slice_indices(mat.shape, i) + [torch.tensor(range(b[i][0]))]] = value
        mat[get_slice_indices(mat.shape, i) + [torch.tensor(range(mat.shape[i] - b[i][1], mat.shape[i]))]] = value
    return mat

def get_tensor_slice(tensor, dims, sizes):
    """Given dims and corresponding sizes, get a tensor slice
    When sizes is a list of integers, return a randomly sliced tensor;
    when sizes is a list of (start, end), return a sliced tensor matching the dims and sizes
    
    Example:
        tensor = torch.randn(3, 20, 30)
        assert get_tensor_slice(tensor, dims=[1, 2], sizes=[5, 7]).shape == torch.Size([3, 5, 7])
        assert torch.norm(get_tensor_slice(tensor, dims=[1, 2], sizes=[(5, 13), (8, 22)]) - tensor[:, 5:13, 8:22]) == 0
    """
    shape = tensor.shape
    indices = get_slice_indices(shape, dim=tensor.ndim-1)
    indices = indices + [torch.tensor(range(shape[-1]))]
    for dim, size in zip(dims, sizes):
        if isinstance(size, int):
            start = np.random.choice(shape[dim]-size+1)
            end = start + size
        else:
            start, end = size
        indices[dim] = indices[dim][range(start, end)]
    return tensor[indices]

def cosine_similarity(a, b, dim=0):
    with torch.no_grad():
        a = a - a.mean(dim, keepdim=True)
        b = b - b.mean(dim, keepdim=True)
        cor = nn.functional.cosine_similarity(a, b, dim=dim)
    return cor


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
 
def get_slice(shift, length=None, both_sides=False, multi_dim=False):
    if isinstance(shift, (list, tuple)) and multi_dim:
        if length is None or isinstance(length, int):
            length = [length] * len(shift)
        assert len(shift) == len(length)
        return tuple([get_slice(s, l, both_sides=both_sides, multi_dim=False) for s, l in zip(shift, length)])
    if both_sides:
        if isinstance(shift, int):
            assert shift >=0
            if isinstance(length, int):
                assert shift <= length-shift
                return slice(shift, length-shift)
            else:
                return slice(shift, -shift)
        elif isinstance(shift, (list, tuple)):
            assert len(shift)==2 or len(shift)==3
            return slice(*shift)
        else:
            raise ValueError(f'shift should be an int, list or tuple, but is {shift}')
    if shift == 0:
        return slice(length)
    elif shift < 0:
        return slice(length+shift)
    else:
        return slice(shift, length)

def get_cor(mat, shift_time=0, padding=2, dim=0, func=cosine_similarity, padding_mode='constant'):
    """
    """
    shift_time = np.abs(shift_time)
    shape = mat.shape
    ndim = mat.ndim
    dim = (dim+ndim)%ndim
    if isinstance(padding, int):
        padding = [padding]*ndim*2
        padding[(ndim-dim-1)*2] = 0 # padding from last dimension to first dimension
        padding[(ndim-dim-1)*2+1] = 0
    mat = nn.functional.pad(mat, pad=padding, mode=padding_mode)
    padding = np.array(padding).reshape(ndim, 2)[::-1] # reorder padding from first to last dimension

    shifts = [] 
    for d in range(ndim):
        if d==dim:
            shifts.append([shift_time])
        else:
            shifts.append(range(-padding[d, 0], padding[d, 1] + 1))
    shifts = list(itertools.product(*shifts))
    tmp = [0]*ndim
    tmp[dim] = shift_time
    shifts.remove(tuple(tmp))

    cor = {}
    for s in shifts:
        left = get_slice(
            [(padding[i, 0] + s[i], padding[i, 0] + s[i] + shape[i]) if i!=dim else (s[dim], shape[dim]) 
             for i in range(ndim)], both_sides=True, multi_dim=True)
        left = mat[left]

        right = get_slice(
            [(padding[i, 0], padding[i, 0] + shape[i]) if i!=dim else (0, shape[dim]-s[dim]) 
             for i in range(ndim)], both_sides=True, multi_dim=True)
        right = mat[right]
        cor[s] = func(left, right, dim=dim)
    return cor

# def get_cor__obsolete(mat, shift_time=0, padding=2, dim=0, padding_mode='constant'):
#     """Assume mat has shape (T, nrow, ncol) (dim = 0) 
#     # todo: dim != 0
#     """
#     T = mat.shape[dim]
#     t = [get_slice(shift_time, T), get_slice(-shift_time, T)]
#     shifts = [(0, i) for i in range(1, padding+1)]
#     for i in range(1, padding+1):
#         for j in range(-padding, padding+1):
#             shifts.append((i, j))
#     mat = nn.functional.pad(
#         nn.functional.pad(mat, pad=(padding, padding), mode=padding_mode).transpose(-1, -2), 
#         pad=(padding, padding), mode=padding_mode).transpose(-1, -2)
#     _, nrow, ncol = mat.shape
#     cor = {}
#     for i, s in enumerate(shifts):
#         s1 = [get_slice(s[0], nrow), get_slice(-s[0], nrow)]
#         s2 = [get_slice(s[1], ncol), get_slice(-s[1], ncol)]
#         cor_ = cosine_similarity(mat[t[0], s1[0], s2[0]], mat[t[1], s1[1], s2[1]], dim=0)
#         nrow_, ncol_ = cor_.shape
#         s1_ = [get_slice(s[0], nrow_), get_slice(-s[0], nrow_)]
#         s2_ = [get_slice(s[1], ncol_), get_slice(-s[1], ncol_)]
#         cropping = [get_slice(padding-abs(s[0]), nrow-2*abs(s[0]), both_sides=True), 
#                     get_slice(padding-abs(s[1]), ncol-2*abs(s[1]), both_sides=True)]
#         cor[(s[0], s[1])] = cor_[s1_[0], s2_[0]][cropping[0], cropping[1]]
#         cor[(-s[0], -s[1])] = cor_[s1_[1], s2_[1]][cropping[0], cropping[1]]
#     return cor

def get_cor_map(mat, padding=2, topk=5, dim=0, padding_mode='constant', shift_times=[0,1,2], return_all=False):
    cor_map = []
    for shift_time in shift_times:
        cor = get_cor(mat.float(), shift_time=shift_time, dim=dim, padding=padding, padding_mode=padding_mode)
        torch.cuda.empty_cache()
        cor = torch.stack([v for k, v in sorted(cor.items())])
        cor_map.append(cor)
    cor_map = torch.stack(cor_map, dim=0)
    if return_all:
        return cor_map
    else:
        cor = cor_map.max(dim=0)[0]
        cor_map = cor.topk(topk, dim=0)[0].mean(0)
    return cor_map

def get_cor_map_4d(mat, top_cor_map_percentage=20, padding=2, topk=5, shift_times=[0, 1, 2], select_frames=True, return_all=False, plot=False):
    """mat is a 4-D tensor with shape: (num_episodes, frames_per_episode, nrow, ncol)
    if top_cor_map_percentage is False or 0, then use conventional scheme;
    if top_cor_map_percentage is number in (0, 100], 
        then put all correlation maps together and choose the top_cor_map_percentage ones to calculate mean
    """
    if mat.ndim == 3:
        mat = mat.unsqueeze(0)
    num_episodes, frames_per_episode, nrow, ncol = mat.shape
    if select_frames:
        frame_mean = mat.mean((0,2,3)).cpu().numpy()
        frame_mean_threshold = threshold_otsu(frame_mean)
        idx = frame_mean > frame_mean_threshold # only use a subset of frames to calculate correlation
        if plot:
            plt.plot(frame_mean, 'ro-', markersize=5)
            plt.axhline(y=frame_mean_threshold, linestyle='--')
            plt.title('Select frames with mean intensity larger than a threshold')
            plt.show()
    else:
        idx = range(frames_per_episode)
    cor_map_all = torch.stack([get_cor_map(m[idx], padding=padding, topk=topk, shift_times=shift_times, return_all=top_cor_map_percentage) 
                           for m in mat], dim=0)     
    if top_cor_map_percentage:
        cor_map = cor_map_all.view(-1, nrow, ncol)
        cor_map = cor_map.topk(int(cor_map.shape[0]*top_cor_map_percentage/100.), dim=0)[0].mean(0)
    else:
        cor_map = cor_map_all.topk(max(int(cor_map.shape[0]//2), 1), dim=0)[0].mean(0)
    if return_all:
        return cor_map, cor_map_all
    else:
        return cor_map

def knn_pool(mat, padding=2, topk=5, padding_mode='constant', normalization='softmax'):
    if topk is None:
        topk = padding*(padding+1)
    nframe, nrow, ncol = mat.shape
    cor = get_cor(mat, padding=padding, padding_mode=padding_mode, shift_time=0)
    torch.cuda.empty_cache()
    cor[(0, 0)] = cor[(0, 1)].new_ones(cor[(0, 1)].shape)
    cor = torch.stack([v for k, v in sorted(cor.items())], dim=0)
    attention_unnormalized, order = cor.topk(topk, dim=0)
    if normalization == 'softmax':
        attention = nn.functional.softmax(attention_unnormalized, dim=0)
    elif normalization == 'linear':
        attention_unnormalized = nn.functional.relu(attention_unnormalized, inplace=True)
        attention = attention_unnormalized / attention_unnormalized.sum(dim=0, keepdim=True)
    elif normalization == 'uniform':
        attention = attention_unnormalized.new_ones(attention_unnormalized.shape)/topk
    else:
        raise ValueError(f'normalization = {normalization} unhandled')

    mat = nn.functional.pad(
        nn.functional.pad(mat, pad=(padding, padding), mode=padding_mode).transpose(-1, -2), 
        pad=(padding, padding), mode=padding_mode).transpose(-1, -2)
    if mat.numel() < 6*10**7:
        new_mat = torch.stack([torch.stack([mat[:, i:i+2*padding+1, j:j+2*padding+1].reshape(nframe, -1) for j in range(ncol)], dim=-1) 
                     for i in range(nrow)], dim=-2)
        new_mat = (new_mat[:, order, torch.tensor(range(nrow)).unsqueeze(-1), torch.tensor(range(ncol))] * attention).sum(1)
    else:
        new_mat = []
        for i in range(nrow):
            for j in range(ncol):
                m = mat[:, i:i+2*padding+1, j:j+2*padding+1].reshape(nframe, -1)
                new_mat.append(torch.mv(m[:, order[:, i, j]], attention[:, i, j]))
        new_mat = torch.stack(new_mat, dim=1).reshape(nframe, nrow, ncol)
    del mat, cor, attention_unnormalized, order, attention
    torch.cuda.empty_cache()
    return new_mat

def get_local_mean(mat, kernel_size=5, padding=2, stride=1, padding_mode='zeros'):
    if mat.ndim == 1:
        mat = mat.unsqueeze(0).unsqueeze(1)
    if mat.ndim == 2:
        mat = mat.unsqueeze(1)
    in_channels = mat.size(1)
    if mat.ndim == 3:
        Conv = nn.Conv1d
    elif mat.ndim == 4:
        Conv = nn.Conv2d
    elif mat.ndim == 5:
        Conv = nn.Conv3d
    else:
        raise ValueError(f'mat.ndim={mat.ndim} not handled!')
    model = Conv(in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                 groups=in_channels, bias=False, padding_mode=padding_mode).to(mat.device)
    model.weight.data.fill_(1.)
    with torch.no_grad():
        numerator = model(mat)
        denominator = model(mat.new_ones(mat.shape)) 
        y = numerator / denominator # a novel way to handle boundaries
    return y


def get_local_median(tensor, window_size=50, dim=-1):
    n = tensor.size(dim)
    median = []
    with torch.no_grad():
        for i in range(n):
            index = torch.tensor(range(max(0, i-window_size//2), min(n, i+window_size//2+1)), device=tensor.device)
            median.append(tensor.index_select(dim=dim, index=index).median(dim=dim)[0])
    median = torch.stack(median, dim=dim)
    return median