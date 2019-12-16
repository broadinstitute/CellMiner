import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import closing, square
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.color import label2rgb
from scipy.spatial import ConvexHull

import torch
import torch.nn as nn


def plot_curves(xs, colors, labels, figsize=(20, 10), show=True, **kwargs):
    plt.figure(figsize=figsize)
    for x, c, label in zip(xs, colors, labels):
        plt.plot(x, c=c, label=label, **kwargs)
    plt.legend()
    if show:
        plt.show()
    

def plot_cdf(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.sort(x.reshape(-1))
    y = np.linspace(0, 1, len(x))
    plt.plot(x, y, 'ro')


def imshow(tensor, dim=None, title='', figsize=(20, 10), show_colorbar=True, **kwargs):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.squeeze().detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=figsize)
    if dim is None:
        im = plt.imshow(tensor, **kwargs)
    else:
        im = plt.imshow(tensor.mean(dim), **kwargs)
    if title != '':
        plt.title(title)
    if show_colorbar:
        fig.colorbar(im)
    plt.show()


def plot_image_label_overlay(image, label_image, sel_idx=None, regions=None, figsize=(20, 10), show=True, title=None, save_file=None):
    """Plot image_label_overlay
    
    Args:
        image: 2-D numpy array or torch.Tensor
        label_image: same size as image, but storing segmentation labels
     
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if regions is None:
        regions = regionprops(label_image)
    image_label_overlay = label2rgb(label_image, image=image)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_label_overlay)
    for i, region in enumerate(regions):
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red' if sel_idx is not None and i==sel_idx else 'green', linewidth=2)
        ax.add_patch(rect)
        ax.text(minc, minr, i+1, color='r')
    ax.set_axis_off()
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    if save_file is not None:
        plt.savefig(save_file)
    if show:
        plt.show()


def plot_tensor(x, y=None, marker='o', figsize=None, title=None, show=False, **kwargs):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy().reshape(-1)
    if figsize is not None:
        plt.figure(figsize=figsize)
    if y is None:
        plt.plot(x, marker, **kwargs)
    else:
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy().reshape(-1)
        plt.plot(x, y, marker, **kwargs)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()

def plot_singular_values(mat, marker='o', figsize=None, title=None, show=True, return_svd=False, use_cpu=False, end=None, **kwargs):
    mat = mat.reshape(len(mat), -1)
    if use_cpu:
        mat = mat.cpu()
    u, s, v = torch.svd(mat)
    plot_tensor(s[:end], marker=marker, figsize=figsize, title=title, show=show, **kwargs)
    if return_svd:
        return u, s, v
        
def plot_hist(x, n_bins=1000, markersize=1, figsize=None, show=False, **kwargs):
    """Plot histogram for torch.Tensor
    
    Args:
        x: torch.Tensor that will be flattened to 1-d
    """
    x = x.detach().reshape(-1)
    plot_tensor(torch.linspace(x.min(), x.max(), n_bins), torch.histc(x, n_bins), markersize=markersize, figsize=figsize, show=show, **kwargs)

    
def plot_trace(submat, label_mask, c='b', linestyle=':', marker='.', label=None, log=False, exp=False, **kwargs):
    """Plot traces 
    
    Args:
        submat: 3D torch.Tensor after submat.squeeze(); (time, height, width)
        label_mask: 2D torch.Tensor; (hight, width)
    """
    submat = submat.squeeze()
    assert submat.dim() == 3, f'submat must have 3 dimensions, not {submat.dim()}'
    trace = (submat*label_mask).reshape(submat.size(0), -1).sum(1) / label_mask.sum()
#     trace = trace - trace.min()
    if log:
        trace = torch.log1p(trace)
    if exp:
        trace = torch.exp(trace) - 1
    plt.plot(trace.detach().cpu().numpy(), c=c, linestyle=linestyle, marker=marker, label=label, **kwargs)
    
    
def plot_traces(traces, image, label_image, colors, figsize=(20, 10)):
    num_labels = len(traces)
    # put all traces together
    fig = plt.figure(figsize=figsize)
    for label_idx in range(1, num_labels+1):
        plt.plot(traces[label_idx-1], c=colors[label_idx%len(colors)], linestyle='--', marker='.', label=label_idx)
        plt.legend()
    plt.show()
    # plot the image_label_overlay
    plot_image_label_overlay(image, label_image, regions=None, figsize=figsize)
    # plot each trace separately
    for i in range(num_labels):
        fig = plt.figure(figsize=figsize) 
        plt.plot(traces[i], c=colors[(i+1)%len(colors)], linestyle=':', marker='.')
        plt.title(i+1)
        plt.show()
        
        
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        
        
def plot_3d_scatter(img, filter_fn=None, s=1, alpha=.1, cmap=plt.get_cmap('cool'), title=None, figsize=None, labels=None, 
                   set_equal_axis=False, set_lim=False, show=True):
    """3D scatter plot
    
    Args:
        img: 3D array or torch.Tensor after squeeze()
        filter_fn: a function handle to filter out which points in the img are to be plotted;
            default None, use all points
    """
    if isinstance(img, torch.Tensor):
        img = img.squeeze().detach().cpu().numpy()
    ndim = img.ndim
    assert ndim == 3
    coords = np.indices(img.shape).reshape(ndim, -1)
    if filter_fn is None:
        sel_idx = range(img.size)
    else:
        sel_idx = (filter_fn(img)).reshape(-1)
    xs = coords[0, sel_idx]
    ys = coords[1, sel_idx]
    zs = coords[2, sel_idx]
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.scatter(xs=xs, ys=ys, zs=zs, c=img.reshape(-1)[sel_idx], s=s, alpha=alpha, cmap=cmap)
    if labels is None:
        labels = ['x', 'y', 'z']
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    if set_lim is True:
        ax.set_xlim([xs.min(), xs.max()])
        ax.set_ylim([ys.min(), ys.max()])
        ax.set_zlim([zs.min(), zs.max()])
    if set_equal_axis:
        axisEqual3D(ax)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()
    
        
def plot_image(image, filter_fn=None, s=1, alpha=.1, cmap=plt.get_cmap('cool'), title=None, labels=None, mean_dim=None, 
               figsize=None, force_3d=False, show=True, **kwargs):
    """Plot 1-d curve, 2-d image, and 3-d scatter plot
    """
    if isinstance(image, torch.Tensor):
        image = image.squeeze().detach().cpu().numpy()
    if mean_dim is not None:
        image = image.mean(mean_dim)
    ndim = image.ndim
    if ndim == 1:
        plt.plot(image, **kwargs)
    elif ndim == 2:
        if force_3d:
            ind0, ind1 = np.indices(image.shape)
            fig = plt.figure(figsize=None)
            ax = fig.gca(projection='3d')
            ax.plot_surface(ind0, ind1, image, cmap=cmap, **kwargs)
        else:
            plt.imshow(image, cmap=cmap, **kwargs)
    elif ndim == 3:
        plot_3d_scatter(image, filter_fn=filter_fn, s=s, alpha=alpha, cmap=cmap, title=None, labels=labels, show=False, **kwargs)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()
        
        
def plot_images(x, model, cmap=plt.get_cmap('cool'), repeat=1, mean_dim=None, target=None, use_same_thresh=False,
               pred_fn=None, pred_fn_args=None):
    """For UNet models which keep the output shape the same as the input, plot multiple 2D images or 3D scatter plots 
    by feeding data to model multiple times
    
    Args: 
        x: torch.Tensor
        model: a Unet model
    """
    img = x.squeeze().detach().cpu().numpy()
    ndim = img.ndim
    assert ndim==1 or ndim == 2 or ndim == 3
    bg_threshold = threshold_otsu(img)
    title = 'Input'
    plot_image(img, cmap=cmap, filter_fn=lambda x: x > bg_threshold, title=title, mean_dim=mean_dim)
    if target is not None:
        plot_image(target, cmap=cmap, filter_fn=lambda x: x > bg_threshold, title='Target', mean_dim=mean_dim)
    if pred_fn is None:
        y_pred = model(x)
    else:
        y_pred = pred_fn(model, x, **pred_fn_args)
    img = y_pred.squeeze().detach().cpu().numpy()
    if not use_same_thresh:
        bg_threshold = threshold_otsu(img)
    title = 'Output - 1'
    plot_image(img, cmap=cmap, filter_fn=lambda x: x > bg_threshold, title=title, mean_dim=mean_dim)
    for i in range(repeat):
        if pred_fn is None:
            y_pred = model(x)
        else:
            y_pred = pred_fn(model, x, **pred_fn_args)
        img = y_pred.squeeze().detach().cpu().numpy()
        if not use_same_thresh:
            bg_threshold = threshold_otsu(img)
        title = f'Output - {i+2}'
        plot_image(img, cmap=cmap, filter_fn=lambda x: x > bg_threshold, title=title, mean_dim=mean_dim)
        

def get_ellipse(coords, center, radius):
    """Get the indices of ellipses
    
    Args:
        coords: 2-d array with shape=(N, dim), where N is the number of `ellipses` to be generated, and dim is the dimension
        center: 1-d array with shape=(dim, )
        radius: 1-d array with shape=(dim, )
        
    Returns:
        
    """
    s = np.array([np.square(x - c) / r**2 for x, c, r in zip(coords, center, radius)]).sum(0)
    return s <= 1


def in_hull(p, hull):
    return all(hull.equations[:, :-1].dot(p) + hull.equations[:, -1] <= 0)


def get_image(size=None, centers=None, radii=None, image=None, values_ellipse=None, lows=None, highs=None, n_pts=None,
              values_hull=None, plot=False, return_locations=False, cmap=plt.get_cmap('cool'), additive=False):
    """Generate a new 'image' (multi-dimensional array)
    
    Args:
        size: 
        centers: for ellipses
        radii: for ellipses
        image: if not None, perform in-place operation
        values_ellipse: for ellipses
        lows:
        highs:
        npts:
        values_hull:
        
    """
    if image is None:
        image = np.zeros(size)
    else:
        size = image.shape
    ndim = len(size)
    coords = np.indices(image.shape).reshape(ndim, -1)
    locations = []
    if centers is not None and radii is not None:
        if values_ellipse is None:
            values_ellipse = [1] * len(centers)
        for center, radius, value in zip(centers, radii, values_ellipse):
            sel_idx = get_ellipse(coords, center, radius)
            sel_loc = sel_idx.reshape(size)
            if additive:
                image[sel_loc] += value
            else:
                image[sel_loc] = value
            locations.append(np.nonzero(sel_loc))
        
    if lows is not None and highs is not None and n_pts is not None:
        if values_hull is None:
            values_hull = [1] * len(lows)
        for low, high, n, value in zip(lows, highs, n_pts, values_hull):
            points = np.random.randint(low, high, (n, ndim))
            hull = ConvexHull(points)
            sel_loc = np.array([in_hull(p, hull) for p in coords.T]).reshape(size)
            if additive:
                image[sel_loc] += value
            else:
                image[sel_loc] = value
            locations.append(np.nonzero(sel_loc))
    if plot:
        assert len(image.shape) <= 3
        min_val = min(image.max() if values_ellipse is None else min(values_ellipse), image.max() if values_hull is None else min(values_hull))
        plot_image(image, cmap=cmap, filter_fn=lambda x: x>=min_val)
    if return_locations:
        return image, locations
    return image


def get_lines(num_points, size=None, points=None, image=None, tolerance=1e-4, value=1, additive=True, plot=False):
    """Get lines in an image
    
    Args:
        num_points: int
            the first point will be chosen as the center
        size: tuple or int, ignored if image is not None
        points: np.array with shape (num_points, ndim) or None
        image: n-d array
        tolerance: float, cosine similarity should be approximately 1
    
    Returns:
        image: n-d array
    """
    if image is None:
        image = np.zeros(size)
    size = image.shape
    ndim = len(size)
    if points is None:
        points = np.stack([np.random.randint(0, size[i], (num_points, )) for i in range(ndim)], axis=1)
    points = np.array(points)
    directions = points[1:] - points[:1]
    coordinates = np.indices(size).reshape(ndim, -1).T
    lines = (coordinates - points[:1])
    d1 = np.linalg.norm(lines, axis=-1)
    idx = d1 > 0
    coordinates = coordinates[idx]
    lines = lines[idx]
    d1 = d1[idx, None]
    d2 = np.linalg.norm(directions, axis=-1)
    idx = d2 > 0
    directions = directions[idx]
    d2 = d2[idx]
    sel_idx = np.logical_and((1 - (lines.dot(directions.T) / d1 / d2)) < tolerance, d1 <= d2).sum(1) > 0
    coordinates = coordinates[sel_idx]
    if additive:
        image[tuple([coordinates[:,i] for i in range(ndim)])] += value
    else:
        image[tuple([coordinates[:,i] for i in range(ndim)])] = value
    if plot:
        plot_image(image)
    return image

        
def make_video(array, num_frames=None, interval=20, start_idx=0, cmap='viridis', repeat=True, show_title=False, title=None):
    """Make 2D videos for 2D arrays
    
    Arguments:
        array: 3D tensor or array, the first dimension corresponds to time
        num_frames: array[start_idx:start_idx+num_frames] will be used
        interval: ms between two adjacent frames
        repeat: pass it to animation.FuncAnimation
        
    Returns:
        vid: html5 video handle
    """
    if isinstance(array, torch.Tensor):
        array = array.squeeze().detach().cpu().numpy()
    assert array.ndim == 3
    if num_frames is None:
        num_frames = array.shape[0]
    fig, ax = plt.subplots()
    img = plt.imshow(array[start_idx], cmap=cmap)
    def animate(i, start=start_idx, show_title=show_title):
        img.set_data(array[start_idx+i])
        if show_title:
            if title is not None:
                ax.set_title(title[i])
            else:
                ax.set_title(i)
        return (img, )
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=interval, repeat=repeat)
    vid = anim.to_html5_video()
    return vid


def make_3d_video(data, interval=100, zlim=None, title='', cmap=cm.coolwarm, repeat=True):
    """Make 3D video of data
    
    Arguments:
        data: 3D tensor or array
        interval: ms between two adjacent frames
        
    Returns:
        vid: html5 video handle
        
    """
    if isinstance(data, torch.Tensor):
        data = data.squeeze().detach().cpu().numpy()
    assert data.ndim == 3
    length, nrow, ncol = data.shape
    ind0, ind1 = np.indices([nrow, ncol])

    def update_plot(num, data, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(X=ind0, Y=ind1, Z=data[num], cmap=cmap, linewidth=0, antialiased=False)
        
    fig = plt.figure()
    ax = Axes3D(fig)
    plot = [ax.plot_surface(X=ind0, Y=ind1, Z=data[0], cmap=cmap, linewidth=0, antialiased=False)]
    # Setting the axes properties
    # ax.set_xlim3d([0.0, 1.0])
    ax.set_xlabel('X')
    # ax.set_ylim3d([0.0, 1.0])
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if zlim is None:
        zlim = [data.min(), data.max()]
    ax.set_zlim3d(zlim)
    if title is not None and title != '':
        ax.set_title(title)
    ani = animation.FuncAnimation(fig, update_plot, length, fargs=(data, plot),
                                       interval=interval, blit=False, repeat=repeat)
    vid = ani.to_html5_video()
    return vid