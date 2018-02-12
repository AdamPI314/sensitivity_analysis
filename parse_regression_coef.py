import pandas as pd
import numpy as np
import re
import os
import fit_least_square_regression as flsr


class parse_regression_coef_c:
    # read pathway data as pandas frame object
    @staticmethod
    def read_pathway_as_pandas_frame_object(filename):
        return pd.read_csv(filename, delimiter="\t", names=["number"])

    # convert pandas frame object to list
    @staticmethod
    def convert_pandas_frame_object_to_list(pathway_data_in):
        # get rid of "#"
        mask = pathway_data_in.number.str.contains("#")
        pathway_data = pathway_data_in[~mask]
        return np.array(pathway_data.number).reshape((-1, 4))

    # parse variance, zeroth_order, first_order and second_order values
    @staticmethod
    def var_zero_first_second(coef_list, ith, in_Nth_order_1st=5, in_Nth_order_2nd=2):
        coef_ith = coef_list[ith]
        var = float(coef_ith[0])
        zeroth = float(coef_ith[1])
        first_t = map(float, re.split(",", coef_ith[2]))
        first = np.array(first_t).reshape((-1, in_Nth_order_1st))
        second_t = map(float, re.split(",", coef_ith[3]))
        second = np.array(second_t).reshape(
            (-1, in_Nth_order_2nd * in_Nth_order_2nd))
        return var, zeroth, first, second

    # parse variance, zeroth_order, first_order and second_order values
    # return zeroth_ordre, first_order and second_order in a list
    @staticmethod
    def var_zero_first_second_in_a_list(coef_list, ith):
        coef_ith = coef_list[ith]
        var = float(coef_ith[0])
        zeroth = float(coef_ith[1])
        first_t = [float(x) for x in re.split(",", coef_ith[2])]
        second_t = [float(x) for x in re.split(",", coef_ith[3])]
        return var, [zeroth] + first_t + second_t

    @staticmethod
    def get_var_zero_first_second_coef(data_dir, s_a_s=None):
        """
        return zero-th order, first order and second order coefficient
        """
        if s_a_s is None:
            return
        f_n_coef = os.path.join(data_dir, "output", "fit_coef.inp")
        var_coef_frame_obj = parse_regression_coef_c.read_pathway_as_pandas_frame_object(
            f_n_coef)
        var_coef_list = parse_regression_coef_c.convert_pandas_frame_object_to_list(
            var_coef_frame_obj)

        var_target, fit_coef = parse_regression_coef_c.var_zero_first_second_in_a_list(
            var_coef_list, 0)

        zero_order_coef, first_order_coef, second_order_coef = \
            flsr.fit_1D_2D_all_c.split_1D_coef_array_static(
                fit_coef, s_a_s['N_variable'], s_a_s['Nth_order_1st'], s_a_s['Nth_order_2nd'])

        return var_target, zero_order_coef, first_order_coef, second_order_coef
