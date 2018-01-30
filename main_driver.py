"""
main driver
"""

import os
import sys
import time
import numpy as np

import global_settings
import my_utility as mu
import fit_least_square_regression as flsr
import parse_regression_coef as prc
import variance_correlation_SI as vcs
import calculate_SI as cs
import sensitivity_plot as splot

if __name__ == '__main__':
    TIME_I = time.time()

    FILE_DIR = os.path.abspath(os.path.join(os.path.realpath(sys.argv[0]), os.pardir, os.pardir, os.pardir))
    print(FILE_DIR)

    S_A_S = global_settings.get_s_a_setting(FILE_DIR)

    #u_norm = mu.read_uncertainty(os.path.join(FILE_DIR, "output", "uncertainties_const.csv"),
    #                             os.path.join(FILE_DIR, "output", "k_global.csv"))
    #target_all = mu.read_target(os.path.join(FILE_DIR, "output", "ign_global.csv"))

    #fit_1D_2D_all = flsr.fit_1D_2D_all_c(u_norm, target_all,
    #                                     S_A_S['N_variable'],
    #                                     S_A_S['Nth_order_1st'],
    #                                     S_A_S['Nth_order_2nd'])
    #f_n_coef = os.path.join(FILE_DIR, "output", "fit_coef.inp")
    #fit_1D_2D_all.coef_s2f(f_n_coef)
    #print("fit DONE!")

    #cs.calculate_SI_s2f(FILE_DIR, S_A_S)
    #print("calculate SI DONE!")

    splot.bar_1D_SI(FILE_DIR)
    print("Plot SI DONE!")
