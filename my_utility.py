"""
my utility
"""
import os
import numpy as np


def read_uncertainty(f_n_1, f_n_2):
    """
    read in uncertainty and normalize
    """
    u_const = np.loadtxt(f_n_1, dtype=float, delimiter=',', comments='#')
    u_random = np.loadtxt(f_n_2, dtype=float, delimiter=',', comments='#')

    # data modification, logendre polynomial is orthonormal in the range of
    # (-1, 1)
    for i, _ in enumerate(u_const):
        # species case when u_const is 1.0
        if u_const[i] == 1.0:
            u_random[:, i] = u_random[:, i] * 0.0
            continue
        u_random[:, i] = (u_random[:, i] - 1 / u_const[i]) \
            / (u_const[i] - 1 / u_const[i])
        u_random[:, i] = u_random[:, i] * 2 - 1
    return u_random


def read_target(f_n):
    """
    read in target
    """
    target = np.loadtxt(f_n, dtype=float, delimiter=',', comments='#')
    return target
