import numbers
import warnings
import time

import numpy as np

import torch
import torch.nn

EPSILON = np.finfo(np.float32).eps.item()
INTEGER_TYPES = (numbers.Integral, np.integer)

def _beta_loss_to_float(beta_loss):
    """Convert string beta_loss to float
    """
    allowed_beta_loss = {'frobenius': 2,
                         'kullback-leibler': 1,
                         'itakura-saito': 0}
    if isinstance(beta_loss, str) and beta_loss in allowed_beta_loss:
        beta_loss = allowed_beta_loss[beta_loss]
    if not isinstance(beta_loss, numbers.Number):
        raise ValueError('Invalid beta_loss parameter: got %r instead '
                         'of one of %r, or a float.' %
                         (beta_loss, allowed_beta_loss.keys()))
    return beta_loss


def _check_string_param(solver, regularization, beta_loss, init):
    allowed_solver = ('mu')
    if solver not in allowed_solver:
        raise ValueError(
            'Invalid solver parameter: got %r instead of one of %r' %
            (solver, allowed_solver))
    allowed_regularization = ('both', 'components', 'transformation', None)
    if regularization not in allowed_regularization:
        raise ValueError(
            'Invalid regularization parameter: got %r instead of one of %r' %
            (regularization, allowed_regularization))
    # 'mu' is the only solver that handles other beta losses than 'frobenius'
    if solver != 'mu' and beta_loss not in (2, 'frobenius'):
        raise ValueError(
            'Invalid beta_loss parameter: solver %r does not handle beta_loss'
            ' = %r' % (solver, beta_loss))
    if solver == 'mu' and init == 'nndsvd':
        warnings.warn("The multiplicative update ('mu') solver cannot update "
                      "zeros present in the initialization, and so leads to "
                      "poorer results when used jointly with init='nndsvd'. "
                      "You may try init='nndsvda' or init='nndsvdar' instead.",
                      UserWarning)

    beta_loss = _beta_loss_to_float(beta_loss)
    return beta_loss


def _beta_divergence(X, W, H, beta, square_root=False):
    """Compute the beta-divergence of X and dot(W, H).
    
    Parameters
    ----------
    X : float or array-like, shape (n_samples, n_features)
    W : float or dense array-like, shape (n_samples, n_components)
    H : float or dense array-like, shape (n_components, n_features)
    beta : float, string in {'frobenius', 'kullback-leibler', 'itakura-saito'}
        Parameter of the beta-divergence.
        If beta == 2, this is half the Frobenius *squared* norm.
        If beta == 1, this is the generalized Kullback-Leibler divergence.
        If beta == 0, this is the Itakura-Saito divergence.
        Else, this is the general beta-divergence.
    square_root : boolean, default False
        If True, return np.sqrt(2 * res)
        For beta == 2, it corresponds to the Frobenius norm.
        
    Returns
    -------
        res : float
            Beta divergence of X and np.dot(X, H)
    """
    beta = _beta_loss_to_float(beta)
    WH = torch.mm(W, H)
    # Frobenius norm
    if beta == 2:
        res = ((X - WH)**2).sum() / 2
        if square_root:
            return torch.sqrt(res * 2)
        else:
            return res

    # do not affect the zeros: here 0 ** (-1) = 0 and not infinity
    indices = X > EPSILON
    X_data = X[indices]
    WH_data = WH[indices]
    # used to avoid division by zero
    WH_data[WH_data == 0] = EPSILON

    # generalized Kullback-Leibler divergence
    if beta == 1:
        # fast and memory efficient computation of np.sum(np.dot(W, H))
        sum_WH = torch.dot(W.sum(0), H.sum(1))
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = torch.dot(X_data, torch.log(div))
        # add full np.sum(np.dot(W, H)) - np.sum(X)
        res += sum_WH - X_data.sum()

    # Itakura-Saito divergence
    elif beta == 0:
        div = X_data / WH_data
        res = div.sum() - div.log().sum() - X.numel()

    # beta-divergence, beta not in (0, 1, 2)
    else:
        sum_WH_beta = (WH ** beta).sum()
        sum_X_WH = torch.dot(X_data, WH_data ** (beta - 1))
        res = (X_data ** beta).sum() + (beta - 1) * sum_WH_beta - beta * sum_X_WH
        res /= beta * (beta - 1)

    if square_root:
        return torch.sqrt(2 * res)
    else:
        return res
    
    
def _compute_regularization(alpha, l1_ratio, regularization):
    """Compute L1 and L2 regularization coefficients for W and H"""
    alpha_H = 0.
    alpha_W = 0.
    if regularization in ('both', 'components'):
        alpha_H = float(alpha)
    if regularization in ('both', 'transformation'):
        alpha_W = float(alpha)

    l1_reg_W = alpha_W * l1_ratio
    l1_reg_H = alpha_H * l1_ratio
    l2_reg_W = alpha_W * (1. - l1_ratio)
    l2_reg_H = alpha_H * (1. - l1_ratio)
    return l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H


def _initialize_nmf(X, n_components, init=None, eps=1e-6, return_eigenvalue=False):
    """Algorithms for NMF initialization.
    
    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.
    n_components : integer
        The number of components desired in the approximation.
    init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
        Method used to initialize the procedure.
        Default: None.
        Valid options:
        - None: 'nndsvd' if n_components <= min(n_samples, n_features),
            otherwise 'random'.
        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)
        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)
        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)
        - 'custom': use custom matrices W and H
    eps : float
        Truncate all values less than this in output to zero.
        
    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Initial guesses for solving X ~= WH
    H : array-like, shape (n_components, n_features)
        Initial guesses for solving X ~= WH
    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    n_samples, n_features = X.shape

    if (init is not None and init != 'random'
            and n_components > min(n_samples, n_features)):
        raise ValueError("init = '{}' can only be used when "
                         "n_components <= min(n_samples, n_features)"
                         .format(init))

    if init is None:
        if n_components <= min(n_samples, n_features):
            init = 'nndsvd'
        else:
            init = 'random'

    # Random initialization
    if init == 'random':
        avg = torch.sqrt(X.mean() / n_components)
        H = avg * torch.randn(n_components, n_features, device=X.device).abs()
        W = avg * torch.randn(n_samples, n_components, device=X.device).abs()
        return W, H

    # NNDSVD initialization
    U, S, V = torch.svd(X)
    W, H = X.new_zeros(n_samples, n_components), X.new_zeros(n_components, n_features)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = torch.sqrt(S[0]) * torch.abs(U[:, 0])
    H[0, :] = torch.sqrt(S[0]) * torch.abs(V[:, 0])

    for j in range(1, n_components):
        x, y = U[:, j], V[:, j]

        # extract positive and negative parts of column vectors
        x_p, y_p = torch.max(x, x.new_full([1], 0)), torch.max(y, y.new_full([1], 0))
        x_n, y_n = -torch.min(x, x.new_full([1], 0)), -torch.min(y, y.new_full([1], 0))

        # and their norms
        x_p_norm, y_p_norm = torch.norm(x_p), torch.norm(y_p)
        x_n_norm, y_n_norm = torch.norm(x_n), torch.norm(y_n)

        m_p, m_n = x_p_norm * y_p_norm, x_n_norm * y_n_norm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_norm
            v = y_p / y_p_norm
            sigma = m_p
        else:
            u = x_n / x_n_norm
            v = y_n / y_n_norm
            sigma = m_n

        lbd = torch.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0
#     W = W * (W>=eps).float()
#     H = H * (H>=eps).float()

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        avg = X.mean()
        W[W == 0] = abs(avg * torch.randn(len(W[W == 0]), device=X.device) / 100)
        H[H == 0] = abs(avg * torch.randn(len(H[H == 0]), device=X.device) / 100)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))

    if return_eigenvalue:
        return W, H, S
    return W, H


def _multiplicative_update_w(X, W, H, beta_loss, l1_reg_W, l2_reg_W, gamma,
                             H_sum=None, HHt=None, XHt=None, update_H=True):
    """update W in Multiplicative Update NMF"""
    if beta_loss == 2:
        # Numerator
        if XHt is None:
            XHt = torch.mm(X, H.T)
        if update_H:
            # avoid a copy of XHt, which will be re-computed (update_H=True)
            numerator = XHt
        else:
            # preserve the XHt, which is not re-computed (update_H=False)
            numerator = XHt.clone()

        # Denominator
        if HHt is None:
            HHt = torch.mm(H, H.T)
        denominator = torch.mm(W, HHt)
    else:
        # Numerator
        # if X is sparse, should compute WH only where X is non zero
        WH_safe_X = torch.mm(W, H)
        WH = WH_safe_X.clone()
        if beta_loss - 1. < 0:
            WH[WH == 0] = EPSILON

        # to avoid taking a negative power of zero
        if beta_loss < 2:
            WH_safe_X[WH_safe_X == 0] = EPSILON

        if beta_loss == 1:
            WH_safe_X = X / WH_safe_X
        elif beta_loss == 0:
            # speeds up computation time
            # refer to /numpy/numpy/issues/9363
            WH_safe_X = 1 / WH_safe_X
            WH_safe_X = WH_safe_X ** 2
            # element-wise multiplication
            WH_safe_X *= X
        else:
            WH_safe_X = WH_safe_X ** (beta_loss - 2)
            # element-wise multiplication
            WH_safe_X *= X

        # here numerator = dot(X * (dot(W, H) ** (beta_loss - 2)), H.T)
        numerator = torch.mm(WH_safe_X, H.T)

        # Denominator
        if beta_loss == 1:
            if H_sum is None:
                H_sum = H.sum(1)  # shape(n_components, )
            denominator = H_sum.unsqueeze(0)
        else:
            WH = WH ** (beta_loss - 1)
            denominator = torch.mm(WH, H.T)

    # Add L1 and L2 regularization
    if l1_reg_W > 0:
        denominator += l1_reg_W
    if l2_reg_W > 0:
        denominator += l2_reg_W * W
    denominator[denominator == 0] = EPSILON
    delta_W = numerator / denominator

    # gamma is in ]0, 1]
    if gamma != 1:
        delta_W = delta_W ** gamma

    return delta_W, H_sum, HHt, XHt


def _multiplicative_update_h(X, W, H, beta_loss, l1_reg_H, l2_reg_H, gamma):
    """update H in Multiplicative Update NMF"""
    if beta_loss == 2:
        numerator = torch.mm(W.T, X)
        denominator = torch.mm(torch.mm(W.T, W), H)

    else:
        # Numerator
        WH_safe_X = torch.mm(W, H)
        # copy used in the Denominator
        WH = WH_safe_X.clone()
        if beta_loss < 1:
            WH[WH == 0] = EPSILON

        # to avoid division by zero
        if beta_loss < 2:
            WH_safe_X[WH_safe_X == 0] = EPSILON

        if beta_loss == 1:
            WH_safe_X = X / WH_safe_X
        elif beta_loss == 0:
            # speeds up computation time
            # refer to /numpy/numpy/issues/9363
            WH_safe_X = 1 / WH_safe_X
            WH_safe_X = WH_safe_X ** 2
            # element-wise multiplication
            WH_safe_X *= X
        else:
            WH_safe_X = WH_safe_X ** (beta_loss - 2)
            # element-wise multiplication
            WH_safe_X *= X

        # here numerator = dot(W.T, (dot(W, H) ** (beta_loss - 2)) * X)
        numerator = torch.mm(W.T, WH_safe_X)

        # Denominator
        if beta_loss == 1:
            W_sum = W.sum(0) # shape(n_components, )
            W_sum[W_sum == 0] = 1.
            denominator = W_sum.unsqueeze(1)

        # beta_loss not in (1, 2)
        else:
            # computation of WtWH = dot(W.T, dot(W, H) ** beta_loss - 1)
            WH = WH ** (beta_loss - 1)
            denominator = torch.mm(W.T, WH)

    # Add L1 and L2 regularization
    if l1_reg_H > 0:
        denominator += l1_reg_H
    if l2_reg_H > 0:
        denominator += l2_reg_H * H
    denominator[denominator == 0] = EPSILON
    delta_H = numerator / denominator

    # gamma is in ]0, 1]
    if gamma != 1:
        delta_H = delta_H ** gamma

    return delta_H


def _fit_multiplicative_update(X, W, H, beta_loss='frobenius',
                               max_iter=200, tol=1e-4,
                               l1_reg_W=0, l1_reg_H=0, l2_reg_W=0, l2_reg_H=0,
                               update_H=True, verbose=0):
    """Compute Non-negative Matrix Factorization with Multiplicative Update
    The objective function is _beta_divergence(X, WH) and is minimized with an
    alternating minimization of W and H. Each minimization is done with a
    Multiplicative Update.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Constant input matrix.
    W : array-like, shape (n_samples, n_components)
        Initial guess for the solution.
    H : array-like, shape (n_components, n_features)
        Initial guess for the solution.
    beta_loss : float or string, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros.
    max_iter : integer, default: 200
        Number of iterations.
    tol : float, default: 1e-4
        Tolerance of the stopping condition.
    l1_reg_W : double, default: 0.
        L1 regularization parameter for W.
    l1_reg_H : double, default: 0.
        L1 regularization parameter for H.
    l2_reg_W : double, default: 0.
        L2 regularization parameter for W.
    l2_reg_H : double, default: 0.
        L2 regularization parameter for H.
    update_H : boolean, default: True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.
    verbose : integer, default: 0
        The verbosity level.
    Returns
    -------
    W : array, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.
    H : array, shape (n_components, n_features)
        Solution to the non-negative least squares problem.
    n_iter : int
        The number of iterations done by the algorithm.
    References
    ----------
    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    """
    start_time = time.time()

    beta_loss = _beta_loss_to_float(beta_loss)

    # gamma for Maximization-Minimization (MM) algorithm [Fevotte 2011]
    if beta_loss < 1:
        gamma = 1. / (2. - beta_loss)
    elif beta_loss > 2:
        gamma = 1. / (beta_loss - 1.)
    else:
        gamma = 1.

    # used for the convergence criterion
    error_at_init = _beta_divergence(X, W, H, beta_loss, square_root=True)
    previous_error = error_at_init

    H_sum, HHt, XHt = None, None, None
    for n_iter in range(1, max_iter + 1):
        # update W
        # H_sum, HHt and XHt are saved and reused if not update_H
        delta_W, H_sum, HHt, XHt = _multiplicative_update_w(
            X, W, H, beta_loss, l1_reg_W, l2_reg_W, gamma,
            H_sum, HHt, XHt, update_H)
        W *= delta_W

        # necessary for stability with beta_loss < 1
        if beta_loss < 1:
            W[W < EPSILON] = 0.

        # update H
        if update_H:
            delta_H = _multiplicative_update_h(X, W, H, beta_loss, l1_reg_H,
                                               l2_reg_H, gamma)
            H *= delta_H

            # These values will be recomputed since H changed
            H_sum, HHt, XHt = None, None, None

            # necessary for stability with beta_loss < 1
            if beta_loss <= 1:
                H[H < EPSILON] = 0.

        # test convergence criterion every 10 iterations
        if tol > 0 and n_iter % 10 == 0:
            error = _beta_divergence(X, W, H, beta_loss, square_root=True)

            if verbose:
                iter_time = time.time()
                print("Epoch %02d reached after %.3f seconds, error: %f" %
                      (n_iter, iter_time - start_time, error))

            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print("Epoch %02d reached after %.3f seconds." %
              (n_iter, end_time - start_time))

    return W, H, n_iter


def non_negative_factorization(X, W=None, H=None, n_components=None,
                               init='nndsvd', update_H=True, solver='mu',
                               beta_loss='frobenius', tol=1e-4,
                               max_iter=2000, alpha=1., l1_ratio=0.5,
                               regularization='both', verbose=0, return_eigenvalue=False, eigenvalue_threshold=0.9, shuffle=False):
    r"""Compute Non-negative Matrix Factorization (NMF)
    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.
    The objective function is::
        0.5 * ||X - WH||_Fro^2
        + alpha * l1_ratio * ||vec(W)||_1
        + alpha * l1_ratio * ||vec(H)||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
        + 0.5 * alpha * (1 - l1_ratio) * ||H||_Fro^2
    Where::
        ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
        ||vec(A)||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)
    For multiplicative-update ('mu') solver, the Frobenius norm
    (0.5 * ||X - WH||_Fro^2) can be changed into another beta-divergence loss,
    by changing the beta_loss parameter.
    The objective function is minimized with an alternating minimization of W
    and H. If H is given and update_H=False, it solves for W only.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Constant matrix.
    W : array-like, shape (n_samples, n_components)
        If init='custom', it is used as initial guess for the solution.
    H : array-like, shape (n_components, n_features)
        If init='custom', it is used as initial guess for the solution.
        If update_H=False, it is used as a constant, to solve for W only.
    n_components : integer
        Number of components, if n_components is not set all features
        are kept.
    init : None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar' | 'custom'
        Method used to initialize the procedure.
        Default: 'random'.
        The default value will change from 'random' to None in version 0.23
        to make it consistent with decomposition.NMF.
        Valid options:
        - None: 'nndsvd' if n_components < n_features, otherwise 'random'.
        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)
        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)
        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)
        - 'custom': use custom matrices W and H
    update_H : boolean, default: True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.
    solver : 'cd' | 'mu'
        Numerical solver to use:
        'cd' is a Coordinate Descent solver that uses Fast Hierarchical
            Alternating Least Squares (Fast HALS).
        'mu' is a Multiplicative Update solver.
        .. versionadded:: 0.17
           Coordinate Descent solver.
        .. versionadded:: 0.19
           Multiplicative Update solver.
    beta_loss : float or string, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.
        .. versionadded:: 0.19
    tol : float, default: 1e-4
        Tolerance of the stopping condition.
    max_iter : integer, default: 200
        Maximum number of iterations before timing out.
    alpha : double, default: 0.
        Constant that multiplies the regularization terms.
    l1_ratio : double, default: 0.
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    regularization : 'both' | 'components' | 'transformation' | None
        Select whether the regularization affects the components (H), the
        transformation (W), both or none of them.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : integer, default: 0
        The verbosity level.
    shuffle : boolean, default: False
        If true, randomize the order of coordinates in the CD solver.
    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.
    H : array-like, shape (n_components, n_features)
        Solution to the non-negative least squares problem.
    n_iter : int
        Actual number of iterations.
        
    Examples
    --------
    nrow, ncol, L = 180, 300, 3000
    d = nrow*ncol
    T = L
    u = torch.randn(d,1,device=device).abs()
    v = torch.randn(T, device=device).abs()
    R = u*v
    size = (nrow, ncol)
    U, V, n_iter = non_negative_factorization(R, init=None, return_eigenvalue=True)
    torch.norm(R - U*V) / torch.norm(R)
    
    References
    ----------
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.
    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    """
    beta_loss = _check_string_param(solver, regularization, beta_loss, init)

    if X.min() == 0 and beta_loss <= 0:
        raise ValueError("When beta_loss <= 0 and X contains zeros, "
                         "the solver may diverge. Please add small values to "
                         "X, or use a positive beta_loss.")

    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, INTEGER_TYPES) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)

    if init == "warn":
        init = "random"
    # check W and H, or initialize them
    if init == 'custom' and update_H:
        assert H.shape == (n_components, n_features)
        assert W.shape == (n_samples, n_components)
    elif not update_H:
        assert H.shape == (n_components, n_features)
        # 'mu' solver should not be initialized by zeros
        if solver == 'mu':
            avg = torch.sqrt(X.mean() / n_components)
            W = X.new_full((n_samples, n_components), avg)
        else:
            W = X.new_zeros((n_samples, n_components))
    else:
        init_w_h = _initialize_nmf(X, n_components, init=init, return_eigenvalue=return_eigenvalue)
        if len(init_w_h) == 3:
            W, H, S = init_w_h
            num_components = torch.nonzero(S.cumsum(dim=0)/S.sum() >= eigenvalue_threshold).min().item() + 1
            W = W[:, :num_components]
            H = H[:num_components]
        elif len(init_w_h) == 2:
            W, H = init_w_h

    l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = _compute_regularization(alpha, l1_ratio, regularization)

    if solver == 'mu':
        W, H, n_iter = _fit_multiplicative_update(X, W, H, beta_loss, max_iter,
                                                  tol, l1_reg_W, l1_reg_H,
                                                  l2_reg_W, l2_reg_H, update_H,
                                                  verbose)

    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)

    if n_iter == max_iter and tol > 0:
        warnings.warn("Maximum number of iteration %d reached. Increase it to"
                      " improve convergence." % max_iter)
    return W, H, n_iter


def nmf_naive(V, num_components, W=None, H=None, tol=1e-3, lr=1, max_num_iter=1000, random_init=True):
    n, m = V.shape
    if W is None:
        if random_init:
            W = torch.randn(n, num_components).abs()
        else:
            W = V.mean(dim=1).repeat(num_components).reshape([n, num_components])
    if H is None:
        if random_init:
            H = torch.randn(num_components, m).abs()
        else:
            H = V.mean(dim=0).repeat(num_components).reshape([num_components, m])
            H = H * V.sum() / (W.sum() * H.sum()) * num_components
    W.requires_grad_()
    H.requires_grad_()
    optimizer = torch.optim.AdamW([W, H], lr=lr)
    for i in range(max_num_iter):
        loss = Beta_divergence(torch.matmul(W, H), V)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(i, loss.item())
        v = torch.matmul(W, H)
        diff = torch.norm(V-v)
        W = W * torch.matmul(V, H.T) / torch.matmul(v, H.T)
        H = H * torch.matmul(W.T, V) / torch.matmul(W.T, v)
        if diff <= tol:
            break
        print(i, diff.item())
    return W, H