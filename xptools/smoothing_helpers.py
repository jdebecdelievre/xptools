import scipy.sparse as sp
from joblib import Parallel, delayed
import pdb
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from cvxpy.error import SolverError

def hampel_filter(data_orig, k=7, t0=3, sigma_noise=0.02, replace_by_median=False):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    sigma_noise: noise already present in the data
    replace_by_median: Whether to replace the outliers by the local median or by NaN
    source: https://stackoverflow.com/questions/46819260/filtering-outliers-how-to-make-median-based-hampel-function-faster
    '''

    #Make copy so original not edited
    data = data_orig.copy()

    #Hampel Filter
    L = 1.4826
    rolling_median = data.rolling(window=k, center=True, min_periods=int((k+1)/2)).apply(np.nanmedian, raw=True)
    # rolling_median = data.rolling(window=k, center=True).median()
    difference = np.abs(data - rolling_median)

    MAD = lambda x: np.nanmedian(np.abs(x - np.nanmedian(x)))
    rolling_MAD = data.rolling(window=k, center=True, min_periods=int((k+1)/2)).apply(MAD, raw=True)
    threshold = t0 * L * rolling_MAD + sigma_noise

    '''
    Perhaps a condition should be added here in the case that the threshold value
    is 0.0; maybe do not mark as outlier. MAD may be 0.0 without the original values
    being equal. See differences between MAD vs SDV.
    '''

    outlier_idx = (difference > threshold).any(axis=1)
    if replace_by_median:
        # replace by the median if the criterion is not respected
        data[outlier_idx] = rolling_median[outlier_idx]

        # replacing by the median gives pretty poor results at extremity -> use NaN
        data.iloc[0] = np.nan if outlier_idx.iloc[0] else data.iloc[0]
        data.iloc[-1] = np.nan if outlier_idx.iloc[-1] else data.iloc[-1]
    else:
        data[outlier_idx] = np.nan

    return(data)

def FusedLassoFilter(D,smoothness, huber_m=1.0, degree=6, order=3):

    # params
    n = D.shape[0]
    A6 = fd_matrix_parsi(n, degree, order, dt=1.0)
    A3 = fd_matrix_parsi(n, dt=1, degree=3, order=2)

    # Localize NaN
    nonNan = 1.0 - np.isnan(D)
    n_ = np.sum(nonNan)

    # Create scalar optimization variables.
    D_ = np.nan_to_num(D)
    x = cp.Variable(shape=n, value=D_)
    l = cp.Parameter(nonneg=True, value=smoothness)

    # Form objective.
    obj = cp.Minimize(cp.norm1(A6 @ x) + cp.sum_squares(A3 @ x) + l*nonNan@cp.huber((x - D_), M=huber_m))
#     obj = cp.Minimize(cp.norm1(A6 @ x) + l*nonNan@cp.square((x - D_))/n_)
                      
#     con = [nonNan@cp.square((x - D_))/n_ <= l]

    # Form and solve problem.
    prob = cp.Problem(obj)
    prob.solve(warm_start=True)

    # assert(prob.status == "optimal"), "Filtering did not converge"

    return x.value, prob.status


def FusedLassoCrossValidation(D, s_table, huber_m=1.0, nn=10, degree=6, order=3, dt=1.0):
    '''
    For each smoothness parameter in s_table, this function estimates the generalization error. It repeatedly - nn times -
    randomly holds out 20% of the data, performs the fit and evaluates the error in the held out data. The error for one
    smoothness parameter is the average of errors held out.
    '''

    # Set Parameters
    n0 = D.shape[0]
    n = int(.8 * n0)
    D_ = np.nan_to_num(D)
    x = cp.Variable(shape=n0, value=D_)
    l = cp.Parameter(nonneg=True, value=1)

    # Localize NaN
    nonNan = sp.diags(1.0 - np.isnan(D))
    n_0 = np.sum(nonNan)
    n_ = cp.Parameter(nonneg=True, value=n_0)
    kpt = cp.Parameter(shape=n0, value=nonNan@np.ones(n0), nonneg=True)

    # Create optimization problem
    A6 = fd_matrix_parsi(n0, degree, order, dt)
    A3 = fd_matrix_parsi(n0, dt=1, degree=3, order=2)

    obj = cp.Minimize(cp.norm1(A6 @ x) + cp.sum_squares(A3 @ x) + l*kpt @cp.huber(x - D_, M=huber_m))
#     obj = cp.Minimize(cp.norm1(A6 @ x) + l*kpt @cp.square(x - D_)/n_)
#     con = [kpt @cp.square(x - D_) / n_ <= l]
    prob = cp.Problem(obj)

    # Create cross-validation data
    kept_indices = np.zeros((nn, n0))
    fld = np.tile(np.array(range(nn)),n0//nn + 1)[:n0]
    for i in range(nn):
        kept = np.ones(n0)
#         indices = np.sort(np.random.choice(n0 - 2, n0 - n, replace=False)) + 1
#         kept[indices] = 0
#         kept_indices[i] = kept
        kept_indices[i] = (fld == i)

    def evaluate(s, i):
        l.value = s
        kpt.value = nonNan@kept_indices[i]
        n_.value = np.sum(kpt.value)
        try:
            prob.solve(warm_start=True)
        except SolverError:
            print("solver crashed when solving fold {} with smoothness {}. Setting error to NaN.".format(i,s))
            return np.nan
        return (np.logical_not(kept_indices[i]) @ (nonNan @ (x.value - D_)) ** 2)/(n_0 - n_.value)

    error = [0] * len(s_table)
    err = [0] * nn
    for i_s, s in enumerate(s_table):
        for i in range(nn):
            err[i] = evaluate(s, i)
        error[i_s] = sum(err)

    return error


def ParallelCrossValidation(fields, s_table, params, cv_params, files, meta_dir, data, smoothness):
    for field in fields:
        errors_table = np.array(Parallel(n_jobs=5)(delayed(
            FusedLassoCrossValidation)(data[file][field], s_table, params['fused_lasso_filter']['huber_m'][field], cv_params['number_random_folds']) for file in files))
        # FusedLassoCrossValidation(data['trajectory77.csv'][field], s_table, cv['number_random_folds'])
        
        errors = np.sum(errors_table, axis=0)
        smoothness[field] = s_table[np.argmin(errors)]

        # plot mean error versus smoothness parameter
        plt.figure(figsize=(10,6))
        plt.loglog(s_table, errors)
        plt.grid(True)
        plt.xlabel('smoothness parameter')
        plt.ylabel(f'Summed CV error on all trajectories for {field}')
        plt.grid(True)
        plt.scatter(np.argmin(errors), smoothness[field], s=80, facecolors='none', edgecolors='r')
        plt.savefig(os.path.join(meta_dir,'crossvalid_' + field + '.png'))
        plt.close()
        errors[np.isnan(errors)] = 1e10
        

        # Show error for each trajectory
        traj_lbl = [f.split('.')[0] for f in files]
        nt = errors_table.shape[0]
        plt.figure(figsize=(15, 5))
        plt.semilogy(range(nt), errors_table[:, np.argmin(errors)], '+')
        plt.xticks(ticks=np.array(range(nt)), labels=traj_lbl, rotation=45)
        plt.grid(True)
        plt.title(f'CV error on {field} for each trajectory at best smoothness parameter.')
        plt.savefig(os.path.join(meta_dir,'crossvalid_bestFit_' + field + '.png'))
        plt.close()

        # Heat map of errorsop
        
        s_lbl = [f'{s:.2}' for s in s_table]
        fig, ax = plt.subplots(figsize=(len(s_lbl)*1./len(traj_lbl), 5))
        im = ax.imshow(errors_table)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('CV error magnitude', rotation=-90, va="bottom")
        ax.set_xticks(np.arange(len(s_lbl)))
        ax.set_yticks(np.arange(len(traj_lbl)))
        ax.set_xticklabels(s_lbl)
        ax.set_yticklabels(traj_lbl)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(traj_lbl)):
            for j in range(len(s_lbl)):
                text = ax.text(j, i, f'{errors_table[i, j]:.2}',
                            ha="center", va="center", color="w")
        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_title("Heatmap of cross validation errors for each trajectory")
        fig.tight_layout()
        plt.savefig(os.path.join(meta_dir,'crossvalid_heatmap_' + field + '.png'))

        # Save Errors
        errors = pd.DataFrame(data=errors_table.T, columns=[f.split('.')[0] for f in files])
        errors['s_table'] = s_table
        errors.to_csv(os.path.join(meta_dir,'cross_validation_errors_'+field+'.csv'), index=False) 

    return smoothness, errors_table 


def FusedLassoFilterQuat(D,smoothness, degree, order):
    """
    The idea here is to use a relaxation of the quadratically constrained smoothing problem of the quaternions.
     """
    # params
    n = D.shape[0]
    A = fd_matrix_parsi(n, degree, order, dt=1.0)

    # Localize NaN
    nonNan = cp.Constant(sp.diags(1 - np.any(np.isnan(D), axis=1)))

    # Create scalar optimization variables.
    q0 = cp.Variable(shape=n, value=D[:, 0])
    qx = cp.Variable(shape=n, value=D[:, 1])
    qy = cp.Variable(shape=n, value=D[:, 2])
    qz = cp.Variable(shape=n, value=D[:, 3])
    l = cp.Parameter(nonneg=True, value=smoothness)

    # Form objective.
    loss = lambda x, i: nonNan@cp.square((x - D[:, i])) + l*cp.atoms.norm1(A @ x)
    obj = cp.Minimize(loss(q0, 0) + loss(qx, 1) + loss(qy, 2) + loss(qz, 3))
    con = [q0**2 + qx**2 + qy**2 + qz**2 == 1]

    # Form and solve problem.
    prob = cp.Problem(obj, con)
    prob.solve(warm_start=True)

    # assert(prob.status == "optimal"), "Filtering did not converg."

    return np.vstack((q0.value, qx.value, qy.value, qz.value))

def fd_matrix(n, degree, dt):
    """
    Returns an n by n matrix such that, if X is a time serie of length n, A@X will be its derivative.
    A is a full matrix, all the information is used. If n gets big, the linear solver in finite_diff_solve
    starts to spit only very small numbers out. Only use for small n (say < 10).
    """
    M = np.zeros((n,n))
    stencil = np.arange(n)
    for i in range(n):
        M[i,:] = finite_diff_coef(stencil, degree,dt)
        stencil -= 1
    return M


def fd_matrix_parsi(n, degree, order, dt=1.0):
    """
    Returns an n by n matrix such that, if X is a time serie of length n, A@X will be its derivative.
    For the middle of the matrix, central differences are used using "order" points on each side,
    which gives an accuracy of 2*order. On the edges, 2*order+1 points are used as well which gives
    an accuracy equal to 2*order as well.
    dt is 1 by default. Small values of dt may make the matrix get very big.
    """
    # center parts
    stencil = np.arange(-order, order + 1)
    coefs = finite_diff_coef(stencil, degree, dt)
    M = np.eye(n, k=stencil[0]) * coefs[0]
    for i in range(1, len(stencil)):
        M += np.eye(n, k=stencil[i]) * coefs[i]

        # upper columns
    stencil = np.arange(2 * order + 1)
    for i in range(order):
        M[i, :2 * order + 1] = finite_diff_coef(stencil, degree, dt)
        stencil -= 1

    # lower columns
    stencil = np.arange(-2 * order, 1)
    for i in range(order):
        M[n - i - 1, n - 2 * order - 1:n] = finite_diff_coef(stencil, degree, dt)
        stencil += 1
    return sp.csr_matrix(M)


def finite_diff_coef(stencil, degree, dt=1.0):
    '''
    For a given stencil and degree of polynomials, gives the finite difference coefficients.
    Idea and process from http://web.media.mit.edu/~crtaylor/calculator.html
    dt is set to 1 by default, one may want to keep it so to avoid very big numbers
    '''
    mat = np.power(stencil,np.expand_dims(np.arange(0,len(stencil)),1))
    b = np.zeros(len(stencil))
    b[degree] = np.math.factorial(degree)
    return np.linalg.lstsq(mat,b, rcond=None)[0]/ dt**degree