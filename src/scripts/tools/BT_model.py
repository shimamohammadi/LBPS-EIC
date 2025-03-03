from scipy import linalg
import numpy as np
import sys


def BTM(alpha):
    """this code is from zhi li, sureal package"""
    M, M_ = alpha.shape
    assert M == M_

    iteration = 0
    p = 1.0 / M * np.ones(M)
    change = sys.float_info.max

    DELTA_THR = 1e-8

    while change > DELTA_THR:
        iteration += 1
        p_prev = p
        n = alpha + alpha.T
        pp = np.tile(p, (M, 1)) + np.tile(p, (M, 1)).T
        p = np.sum(alpha, axis=1) / np.sum(n / pp, axis=1)

        p = p / np.sum(p)

        change = linalg.norm(p - p_prev)

    pp = np.tile(p, (M, 1)).T + np.tile(p, (M, 1))
    lbda_ii = np.sum(-alpha / np.tile(p, (M, 1)).T**2 + n /
                     pp**2, axis=1)  # summing over axis=1 marginalizes j
    lbda_ij = n / pp*2
    lbda = lbda_ij + np.diag(lbda_ii)
    cova_p = np.linalg.pinv(
        np.vstack([np.hstack([-lbda, np.ones([M, 1])]), np.hstack([np.ones([1, M]), np.array([[0]])])]))
    vari_p = np.diagonal(cova_p)[:-1]
    stdv_p = np.sqrt(vari_p)
    cova_p = cova_p[:-1, :-1]

    cova_v = cova_p / (np.expand_dims(p, axis=1) *
                       (np.expand_dims(p, axis=1).T))
    v = np.log(p)
    stdv_v = stdv_p / p  # y = log(x) -> dy = 1/x * dx

    return v, stdv_v, cova_v, p, stdv_p, cova_p
