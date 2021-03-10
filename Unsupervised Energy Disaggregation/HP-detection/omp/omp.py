from dictionary import norm2
import numpy as np
import scipy.optimize as sci
from tools.result import Result
from tools.rle import rlencode

import time


def OMP(x, Dictionary, dataframe,
        maxiter=1000, ztol=1e-12, tol=1e-10, S=100000,
        threshold_min_power=0):
    """Compute the Orthogonal Matching Pursuit.

    Arguments:
        x -- aggregated load signal
        Dictionary -- Large collection of atoms, each column is an atom
        segment -- Array [start, stop]
        maxiter -- Maximum number of iteration, each iteration the algorithm
                   select an atom 'kj' and update the residual 'Rj'
                   (default=100).
        S -- Sparsity convoyed through a maximum number of coefficient
                (default=1000)
        ztol -- tolerance on the maximum residual covariance allowed,
                i.e. iterations breaks if threshold is achieved
        tol -- convergence tolerance, breaks if the relative error is than
                tol * norm2(x)


    Returns:
        Set of active atoms 'k' with their respective coefficient 'coef'
    """
    # -------------- Check structure args -------------------------------------

    if not type(x).__module__ == np.__name__:  # ifnot numpy matrix
        x = x.values
        x = x[:, 0]  # denesting ndarray

    # -------------- Initialization -------------------------------------------

    xmax = np.max(x)                            # normalizing the signal
    x = x / xmax

    D = Dictionary

    k_index = []                                # vector for the selected atoms
    coef = np.zeros(D.shape[1], dtype=float)    # coefficient vector
    R = x                                       # residual vector for j=1

    xnorm = norm2(x)                            # compute relative err
    err = np.zeros(maxiter, dtype=float)        # relative err vector

    result = Result()

    # ------------- Main interation -------------------------------------------

    for j in range(maxiter):
        # Equation (3) in Arberet et al.
        Rcov = np.dot(D.T, R)       # p-correlation with residual at j=it
        k = np.argmax(Rcov)         # atom indices maximizing p-correlation

        if k not in k_index:        # if the selected atom is not already
            k_index.append(k)       # in the vector, then append it

        # Equation (2) in Arberet et al.
        coefi, _ = sci.nnls(D[:, k_index], x)   # non-negative l.s. solver
        coef[k_index] = coefi

        R = x - np.dot(D[:, k_index], coefi)    # new residual computed
        err[j] = norm2(R)                       # errors for each iteration
        #print(j)

        # TODO Delete certain type of thresholds
        # Stopping criteria :
        resCovk = Rcov[k]
        if resCovk < ztol:
            # print('Stopping criteria: all residual covariances below the threshold')
            break

        if err[j] < tol:
            # print('Stopping criteria: Convergence tolerance achieved')
            break

        if len(k_index) >= S:
            # print('\nLimit on selected atoms achieved')
            break

    # ------------- Additional Atom Selection ---------------------------------

    # Remove the zero coefficients from the support vector
    coef_select = coef[k_index]
    List = list(np.nonzero(coef_select)[0])
    k_index_nonzero = [k_index[i] for i in List]

    # Remove K strongest atom
    k_index_select = []

    for i, ll in enumerate(k_index_nonzero):
        val_rle_output = rlencode(D[:, ll] * coef[ll] * xmax)[2]

        if threshold_min_power < val_rle_output.max():
            k_index_select.append(ll)
        else:
            continue

    # ------------- Preparing Result Outputs ----------------------------------

    signal = x * xmax

    # -------- Output (supplementary) -------- #

    coef_select = coef[k_index_select]
    k_select = D[:, k_index_select]
    RecSignal = np.sum(k_select * coef_select, axis=1) * xmax

    # COMBINED output
    Kcoef = k_select * coef_select * xmax
    nbatoms = k_select.shape[1]
    full_y = signal

    # TODO clear this and delete update ... 

    REFIND = None  # free space for the labels from the clustering method
    optGMMcomponents = None
    ComDict = None  # free space for community detection
    Graph = None
    partition = None

    y_hat_dict = None
    y_truth_dict = None

    # RESULT output update
    result.update(dataframe, signal,
                  maxiter, tol, ztol, S,
                  coef_select, xmax, k_select, RecSignal, k_index_select,
                  err, resCovk,
                  Kcoef, nbatoms, full_y,
                  REFIND, optGMMcomponents,
                  ComDict, Graph, partition,
                  y_hat_dict, y_truth_dict)

    return result
