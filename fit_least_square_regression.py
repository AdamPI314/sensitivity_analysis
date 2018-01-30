"""
fit least square regression
"""
import os
import numpy as np
import scipy.optimize as scopt
import math
import copy


########################################################################################################################
class fit_1D_2D_all_c:
    """
    input: data_all, z, Nth_order=3 (default)
    return: coef2D_all
    """
    # order of legendre polynomial of 1st order coefficient
    Nth_order_1st = 2
    # order of legendre polynomial of 2nd order coefficient
    Nth_order_2nd = 2
    N_variable = 23

    # calculate possible combination
    @staticmethod
    def nCr(n, r):
        func = math.factorial
        return func(n) / func(r) / func(n - r)

    # split 1D coef array into 0th_order, 1st_order and 2nd_order
    @staticmethod
    def split_1D_coef_array_static(my_coef2D, N_variable, Nth_order_1st, Nth_order_2nd):
        N_variable = int(N_variable)
        Nth_order_1st = int(Nth_order_1st)
        Nth_order_2nd = int(Nth_order_2nd)
        # 0th order
        zero_th_order = my_coef2D[0]
        # 1st order; N_variable*Nth_order
        first_order = my_coef2D[1:N_variable * Nth_order_1st + 1]
        coef_1st_order = np.array(first_order).reshape((N_variable, Nth_order_1st))
        # 2nd order; possible combination
        # (factorial(N_variable)/factorial(2)/factorial(N_variable-2))
        second_order = my_coef2D[N_variable * Nth_order_1st + 1:]
        coef_2nd_order = np.array(second_order).reshape((-1, Nth_order_2nd * Nth_order_2nd))
        return zero_th_order, coef_1st_order, coef_2nd_order

    # split 1D coef array into 0th_order, 1st_order and 2nd_order
    def split_1D_coef_array(self):
        # 0th order
        zero_th_order = self.my_coef2D[0]
        # 1st order; N_variable*Nth_order
        first_order = self.my_coef2D[1:self.N_variable * self.Nth_order_1st + 1]
        coef_1st_order = np.array(first_order).reshape((self.N_variable, self.Nth_order_1st))
        # 2nd order; possible combination
        # (factorial(N_variable)/factorial(2)/factorial(N_variable-2))
        second_order = self.my_coef2D[self.N_variable * self.Nth_order_1st + 1:]
        coef_2nd_order = np.array(second_order).reshape((-1, self.Nth_order_2nd * self.Nth_order_2nd))
        return zero_th_order, coef_1st_order, coef_2nd_order

    # The definition below is only for 2D legendre regression
    def my_legendre_polynomial_2D_regression_all(self, data_all, *coef2D_in):
        """
        # input parameter- pointer got to be pointer to 1D array
        # split to 1+23+23 arrays first, const+ 23 1st order arrays+ 23 2nd order arrays
        # reshape 23 2nd order 1D arrays to  2D arrays (rows*columns)
        # There should be just only one constant, but every legendre has a constant term, got to exclude it
        # Every 2D legendre matrix has constant term, 1st order terms, got to exclude them
        """

        # 0th order
        zero_th_order = coef2D_in[0]
        # 1st order; N_variable*Nth_order
        first_order = coef2D_in[1:self.N_variable * self.Nth_order_1st + 1]
        coef_1st_order = np.array(first_order).reshape((self.N_variable, self.Nth_order_1st))
        # 2nd order; possible combination
        # (factorial(N_variable)/factorial(2)/factorial(N_variable-2))
        second_order = coef2D_in[self.N_variable * self.Nth_order_1st + 1:]
        coef_2nd_order = np.array(second_order).reshape((-1, self.Nth_order_2nd * self.Nth_order_2nd))

        # return value
        # 0th_order
        value_t = zero_th_order
        # 1st_order
        for i in range(self.N_variable):
            # insert dummy value, exclude the constant terms
            coef_t = np.insert(coef_1st_order[i, :], 0, 0, axis=0)
            value_t = value_t + \
                np.polynomial.legendre.legval(data_all[:, i], coef_t)
        # 2nd_order
        order_index = 0  # label the 23 matrice
        for i in range(0, self.N_variable):
            for j in range(i + 1, self.N_variable):
                coef_2nd_order_matrix = np.array(coef_2nd_order[order_index, :]).reshape((self.Nth_order_2nd, self.Nth_order_2nd))
                coef_2nd_order_matrix = np.insert(np.insert(coef_2nd_order_matrix, 0, 0, axis=0), 0, 0, axis=1)
                value_t += np.polynomial.legendre.legval2d(data_all[:, i], data_all[:, j], coef_2nd_order_matrix)
                order_index += 1

        return value_t

    def coef_s2f(self, f_n):
        """
        save coefficient to file
        """
        if os.path.isfile(f_n):
            os.remove(f_n)
        zero_order_coef, first_order_coef, second_order_coef = self.split_1D_coef_array()
        with open(f_n, "w") as f_handler:
            f_handler.write("#" + "comments" + "\n")
            f_handler.write("#" + "var" + "\n")
            f_handler.write(str(self.var))
            f_handler.write("\n#" + "zero" + "\n")
            zero_order_coef.tofile(f_handler, sep=", ", format="%s")
            f_handler.write("\n#" + "first" + "\n")
            first_order_coef.tofile(f_handler, sep=", ", format="%s")
            f_handler.write("\n#" + "second" + "\n")
            second_order_coef.tofile(f_handler, sep=", ", format="%s")
            f_handler.write("\n")

    def __init__(self, k_all, target, in_N_variable=8, in_Nth_order_1st=2, in_Nth_order_2nd=2):
        assert in_N_variable == np.shape(k_all)[1]
        self.var = np.var(target)
        self.N_variable = copy.deepcopy(np.shape(k_all)[1])
        self.possible_combination = int(self.nCr(self.N_variable, 2))
        self.Nth_order_1st = copy.deepcopy(int(in_Nth_order_1st))
        self.Nth_order_2nd = copy.deepcopy(int(in_Nth_order_2nd))

        self.index_pair = []
        for i in range(0, self.N_variable):
            for j in range(i + 1, self.N_variable):
                self.index_pair.append((i, j))

        # initial parameters
        self.coef2D_init = np.zeros(1 + self.N_variable * self.Nth_order_1st + self.possible_combination * self.Nth_order_2nd * self.Nth_order_2nd)
        print("length of flatten parameters:\t", len(self.coef2D_init))
        # calculate my_coef, curve_fit with initial parameters
        self.my_coef2D, self.pcov = scopt.curve_fit(self.my_legendre_polynomial_2D_regression_all, k_all, target,
                                                    p0=self.coef2D_init)
