"""
calcualte SI
"""
import os
import sys
import time
import numpy as np

import my_utility as mu
import fit_least_square_regression as flsr
import parse_regression_coef as prc
import variance_correlation_SI as vcs


def calculate_SI_s2f(file_dir, s_a_s=None):
    """
    calculate SI and save to file
    s_a_s represents "sensitivity analyis settings", which is a dictionary
    """
    var_target, _, first_order_coef, second_order_coef = \
        prc.parse_regression_coef_c.get_var_zero_first_second_coef(
            file_dir, s_a_s=s_a_s)

    SI_1st_object = [vcs.SI_1st_c(var_target, first_order_coef[i], np.shape(first_order_coef)[1])
                     for i in range(np.shape(first_order_coef)[0])]
    SI_1st = [x.data for x in SI_1st_object]
    #print(SI_1st, sum(SI_1st))

    SI_2nd_object = [vcs.SI_2nd_c(var_target, second_order_coef[i])
                     for i in range(np.shape(second_order_coef)[0])]
    SI_2nd = [x.data for x in SI_2nd_object]
    #print(SI_2nd, sum(SI_2nd))

    SI_1st_f_n = os.path.join(file_dir, "output", "SI_1st.csv")
    np.savetxt(SI_1st_f_n, SI_1st, fmt='%.18e', delimiter=',')
    SI_2nd_f_n = os.path.join(file_dir, "output", "SI_2nd.csv")
    np.savetxt(SI_2nd_f_n, SI_2nd, fmt='%.18e', delimiter=',')
