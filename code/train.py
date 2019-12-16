import time

import numpy as np

import torch
import torch.nn as nn

from utility import weighted_mse_loss
from visualization import plot_image

def train_model(model, input_fn, input_fn_args={}, num_epoch=30, num_iters=1, print_every=3, 
                optimizer_fn=torch.optim.AdamW, optimizer_fn_args={'lr': 1e-3, 'weight_decay': 1e-2}, 
                pred_fn=None, pred_fn_args={}, loss_fn=weighted_mse_loss, loss_fn_args={}, loss_history_train=[], 
                return_loss_history=False, verbose=True):
    optimizer = optimizer_fn(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_fn_args)
    if pred_fn is None:
        pred_fn = model
    if loss_history_train is None:
        loss_history_train = []
        return_loss_his = True
    start_time = time.time()
    for epoch in range(num_epoch):
        x, y_true = input_fn(**input_fn_args)
        for i in range(num_iters):
            y_pred = pred_fn(x=x, **pred_fn_args)
            loss = loss_fn(y_pred, y_true, **loss_fn_args)
            loss_history_train.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose and (epoch%print_every == 0 or epoch==num_epoch-1):
            print(f'Epoch {epoch+1} batch_avg_loss={np.array(loss_history_train[-num_iters*print_every:]).mean():.2e}')
        torch.cuda.empty_cache()
    end_time = time.time()
    if verbose:
        print(f'Time spent: {end_time-start_time: .1f}s')
    if return_loss_history:
        return loss_history_train


def eval_model(model, input_fn, input_fn_args={}, num_epoch=30, num_iters=1, print_every=3, pred_fn=None, pred_fn_args={}, 
               loss_fn=weighted_mse_loss, loss_fn_args={}, loss_history_test=[], verbose=True, return_loss_history=False, plot=False, cmap=None):
    if pred_fn is None:
        pred_fn = model
    start_time = time.time()
    requires_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    for epoch in range(num_epoch):
        x, y_true = input_fn(**input_fn_args)
        for i in range(num_iters):
            y_pred = pred_fn(x=x, **pred_fn_args)
            loss = loss_fn(y_pred, y_true, **loss_fn_args)
            loss_history_test.append(loss.item())
        if verbose and (epoch%print_every == 0 or epoch==num_epoch-1):
            print(f'Epoch {epoch} batch_avg_loss={np.array(loss_history_test[-num_iters*print_every:]).mean():.2e}')
            if plot:
                plot_image(x.squeeze(), cmap=cmap, title='Input')
                plot_image(y_true.squeeze(), cmap=cmap, title='Target')
                plot_image(y_pred.squeeze(), cmap=cmap, title='Predicted')
        torch.cuda.empty_cache()
    torch.set_grad_enabled(requires_grad)
    end_time = time.time()
    if verbose:
        print(f'Time spent: {end_time-start_time: .1f}s')
    if return_loss_history:
        return loss_history_test


def rank_k_decompose(mat, k=1, max_iter_num=500, eps=1e-1, loss_history=None, random_init_mode=1,
                     optimizer_fn=torch.optim.AdamW, optimizer_fn_args={'lr': 1e-1, 'weight_decay': 1e-2}):
    if random_init_mode==0:
        # NOT useful
        a = mat.sum(dim=-1, keepdim=True)
        b = mat.sum(dim=-2, keepdim=True)
        b = b/b.sum()
    elif random_init_mode==1:
        a = torch.randn(*mat.shape[:-1], k).to(mat.device)
        b = torch.randn(k, mat.shape[-1]).to(mat.device)
    a = a.detach().clone().requires_grad_(True)
    b = b.detach().clone().requires_grad_(True)
    if loss_history is None:
        loss_history = []
    optimizer = optimizer_fn([a, b], **optimizer_fn_args)
    for i in range(max_iter_num):
        loss = torch.norm(mat - torch.matmul(a, b))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if loss < eps or (len(loss_history) > 20 and np.mean(loss_history[-20:-10]) - np.mean(loss_history[-10:]) < eps):
            break
    return a, b, loss_history                          


def step_decompose(mat, num_components=1, max_iter_num=500, eps=1e-3, random_init_mode=1, std_threshold=float('-Inf'),
              optimizer_fn=torch.optim.AdamW, optimizer_fn_args={'lr': 1e-1, 'weight_decay': 1e-2}, verbose=False):
    A = []
    B = []
    loss_history = []
    for i in range(num_components):
        a, b, loss_his = rank_k_decompose(mat, k=1, max_iter_num=max_iter_num, eps=eps, random_init_mode=random_init_mode, 
                                          optimizer_fn=optimizer_fn, optimizer_fn_args=optimizer_fn_args)
        mat = (mat - a*b).detach()
        if verbose:
            print(f'{i:<2} loss={loss_his[-1]:<25} norm={mat.norm():<25} std={mat.std()}')
        A.append(a)
        B.append(b)
        loss_history.append(loss_his)
        if mat.std() < std_threshold:
            break
    return torch.cat(A, dim=-1).detach(), torch.cat(B, dim=-2).detach(), loss_history