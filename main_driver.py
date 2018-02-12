"""
main driver
"""

import os
import sys
import time

import global_settings
import my_utility as mu
import fit_least_square_regression as flsr
import calculate_SI as cs
import sensitivity_plot as splot
import fit_plot as fp
from naming import index_transition

if __name__ == '__main__':
    TIME_I = time.time()

    DATA_DIR = os.path.abspath(os.path.join(os.path.realpath(
        sys.argv[0]), os.pardir, os.pardir, os.pardir, os.pardir, "SOHR_DATA"))
    print(DATA_DIR)

    S_A_S = global_settings.get_s_a_setting(DATA_DIR)
    _, N_2_O_IDX = index_transition(S_A_S['n_dim'], S_A_S['exclude'])

    u_norm = mu.read_uncertainty(os.path.join(DATA_DIR, "output", "uncertainties_const.csv"),
                                 os.path.join(DATA_DIR, "output", "k_global.csv"),
                                 exclude=S_A_S['exclude'])
    target_all = mu.read_target(os.path.join(
        DATA_DIR, "output", "ign_global.csv"))

    if S_A_S['exclude'] is not None:
        S_A_S['N_variable'] -= len(S_A_S['exclude'])
    fit_1D_2D_all = flsr.fit_1D_2D_all_c(u_norm, target_all,
                                         S_A_S['N_variable'],
                                         S_A_S['Nth_order_1st'],
                                         S_A_S['Nth_order_2nd'])
    fit_1D_2D_all.coef_s2f(os.path.join(DATA_DIR, "output", "fit_coef.inp"))
    print("fit DONE!")

    cs.calculate_SI_s2f(DATA_DIR, S_A_S)
    print("calculate SI DONE!")

    splot.bar_1D_SI(DATA_DIR, N_2_O_IDX)
    print("Plot SI DONE!")

    fp.plot_fit_functions(DATA_DIR, S_A_S, N_2_O_IDX)
    print("Plot FIT FUNCTIONS DONE!")
