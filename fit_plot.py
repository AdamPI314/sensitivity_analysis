"""
fit regression plot
"""
import os
import numpy as np

# from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! for plot 3D
import mpl_toolkits.mplot3d.axes3d  # register 3d projection

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
from matplotlib.ticker import FormatStrFormatter

from color_marker import get_colors_markers_linestyles

import my_utility as mu
import parse_regression_coef as prc
import fit_least_square_regression as flsr


class plot_1D_c:
    """
    plot 1D
    save to file
    """
    xmin = -1
    xmax = 1

    @staticmethod
    def func_leg(x, *coef):
        # exclude the 0th order coef, which is a constant
        coef_t = np.insert(coef, 0, 0, axis=0)
        return np.polynomial.legendre.legval(x, coef_t)

    def __init__(self, data_dir, data_sample, target_time, mycoef_1D, zero_order, index, n_2_o_idx=None,
                 file_name="target_vs_K_1D.jpg"):

        label = str(index)
        if n_2_o_idx is not None:
            label = str(n_2_o_idx[int(label)])

        file_name1 = file_name.split(".")[0] + "_" + label + \
            "." + file_name.split(".")[-1]

        colors, markers, _ = get_colors_markers_linestyles()

        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
        x_1 = data_sample[:, index]
        y_1 = target_time
        ax.scatter(x_1, y_1, lw=2, color='#FF7F00',
                   marker=markers[0], label="original")
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

        x_2 = np.linspace(self.xmin, self.xmax, 50)
        y_2 = self.func_leg(x_2, *(mycoef_1D[index, :]))
        ax.plot(
            x_2, y_2, color='r', marker=markers[1], label="1st order fit w/ zero order")

        y_3 = y_2 + zero_order
        ax.plot(
            x_2, y_3, color='blue', marker=markers[2], label="1st order + zero order")

        leg = ax.legend(loc=0, fancybox=True, prop={'size': 10.0})
        leg.get_frame().set_alpha(0.7)
        ax.grid()
        ax.set_xlabel("$k_{" + label + "}$")

        ax.set_ylabel("target")
        ax.set_xlim([self.xmin, self.xmax])

        fig.savefig(os.path.join(data_dir, "output", file_name1), dpi=500)
        plt.close('all')


class plot_2D_c:
    """
    plot 2D
    save to file
    """

    # The definition below is only for 2D legendre value evaluation
    def my_legendre_polynomial_2D_val(self, data_x_y, *coef2D_in):
        # input parameter- pointer got to be pointer to 1D array
        # reshape to a 2D array
        # coef2D = np.reshape(coef2D_in, (self.Nth_order, self.Nth_order))
        coef2D = np.array(coef2D_in).reshape(
            (self.Nth_order_2nd, self.Nth_order_2nd))
        # exclude 0th and 1st order coef
        coef2D = np.insert(np.insert(coef2D, 0, 0, axis=0), 0, 0, axis=1)
        # value
        return np.polynomial.legendre.legval2d(data_x_y[0], data_x_y[1], coef2D)

    def __init__(self, data_dir, data_sample, target_time, my_coef2D, idx_pair_idx, index, n_2_o_idx=None,
                 file_name="target_vs_K_2D.jpg"):
        """
        idx_pair_idx maps index pair (1, 3) to a flatten 1d index
        """
        idx_2d = idx_pair_idx[tuple([index[0], index[1]])]
        self.Nth_order_2nd = int(np.sqrt(len(my_coef2D[idx_2d, :])))

        x_1 = np.linspace(-1, 1)
        y_1 = np.linspace(-1, 1)
        x_g, y_g = np.meshgrid(x_1, y_1)

        fig, a_x = plt.subplots()
        plot1 = a_x.imshow(self.my_legendre_polynomial_2D_val([x_g, y_g], *(my_coef2D[idx_2d, :])),
                           extent=[-1, 1, 1, -1], origin="upper")

        c_b = fig.colorbar(plot1, ax=a_x)
        c_b.formatter.set_scientific(True)
        c_b.formatter.set_powerlimits((-2, 2))
        c_b.update_ticks()

        label0 = str(index[0])
        label1 = str(index[1])
        if n_2_o_idx is not None:
            label0 = str(n_2_o_idx[int(label0)])
            label1 = str(n_2_o_idx[int(label1)])

        a_x.set_xlabel("$k_{" + label0 + "}$")
        a_x.set_ylabel("$k_{" + label1 + "}$")

        a_x.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        a_x.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

        file_name1 = file_name.split(".")[0] + \
            "_" + label0 + "_" + label1 + \
            "_contour." + file_name.split(".")[-1]
        fig.savefig(os.path.join(data_dir, "output", file_name1), dpi=500)

        fig2 = plt.figure()
        a_x_2 = fig2.add_subplot(111, projection='3d')
        plt.hold(True)

        data_1_2 = data_sample[:, index]
        a_x_2.scatter(data_1_2[:, 0], data_1_2[:, 1], target_time,
                      c='#d3ffce', marker='o', s=1.0, alpha=0.8)

        a_x_2.plot_surface(x_g, y_g, self.my_legendre_polynomial_2D_val([x_g, y_g], *(my_coef2D[idx_2d, :])),
                           rstride=1, cstride=1,
                           cmap=plt.cm.get_cmap('jet'), linewidth=0, antialiased=False, alpha=0.8)

        a_x_2.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        a_x_2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        a_x_2.ticklabel_format(axis='z', style='sci', scilimits=(-2, 2))

        a_x_2.set_xlabel("$k_{" + label0 + "}$")
        a_x_2.set_ylabel("$k_{" + label1 + "}$")
        a_x_2.set_zlabel("target")
        a_x_2.set_title("$2^{nd}$ order fit function")
        a_x_2.view_init(37.5, -30)

        # ax2.view_init(20,70)
        # ax2.view_init(100,-30)

        fig2.subplots_adjust(left=0.01, right=0.90, top=0.95)
        file_name2 = file_name.split(".")[0] + "_" + label0 + "_" + label1 + "." + \
            file_name.split(".")[-1]
        fig2.savefig(os.path.join(data_dir, "output", file_name2), dpi=500)
        plt.close('all')


class plot_1D_2D_c:
    """
    plot 1D and 2D together
    save to file
    """
    xmin = -1
    xmax = 1

    # 1D
    def func_leg_1D(self, data_x_y, *zero_first_in):
        # exclude the 0th order coef, which is a constant
        # zero_t= zero_first_in[0]
        coef0 = zero_first_in[1:1 + self.Nth_order_1st];
        coef1 = zero_first_in[1 + self.Nth_order_1st:]
        coef_t0 = np.insert(coef0, 0, 0, axis=0)
        coef_t1 = np.insert(coef1, 0, 0, axis=0)
        return np.polynomial.legendre.legval(data_x_y[0], coef_t0) + np.polynomial.legendre.legval(data_x_y[1], coef_t1)
        # return np.polynomial.legendre.legval(data_x_y[0], coef_t0)+np.polynomial.legendre.legval(data_x_y[1], coef_t1)+zero_t

    # The definition below is only for 2D legendre value evaluation
    def my_legendre_polynomial_2D_val(self, data_x_y, *coef2D_in):
        # input parameter- pointer got to be pointer to 1D array
        # reshape to a 2D array
        # coef2D = np.reshape(coef2D_in, (self.Nth_order_2nd, self.Nth_order_2nd))
        coef2D = np.array(coef2D_in).reshape(
            (self.Nth_order_2nd, self.Nth_order_2nd))
        # exclude 0th and 1st order coef
        coef2D = np.insert(np.insert(coef2D, 0, 0, axis=0), 0, 0, axis=1)
        # value
        return np.polynomial.legendre.legval2d(data_x_y[0], data_x_y[1], coef2D)

    # The definition below is only for 1D and 2D legendre value evaluation
    def my_legendre_polynomial_1D_2D_val(self, data_x_y, *zero_first_second_in):
        # input parameter- pointer got to be pointer to 1D array
        zero_t = zero_first_second_in[0]
        coef0 = zero_first_second_in[1:1 + self.Nth_order_1st]
        coef1 = zero_first_second_in[1 +
                                     self.Nth_order_1st:self.Nth_order_1st + 1 + self.Nth_order_1st]
        coef_t0 = np.insert(coef0, 0, 0, axis=0)
        coef_t1 = np.insert(coef1, 0, 0, axis=0)

        coef2D_in = zero_first_second_in[self.Nth_order_1st +
                                         1 + self.Nth_order_1st:]
        # reshape to a 2D array
        coef2D = np.array(coef2D_in).reshape(
            (self.Nth_order_2nd, self.Nth_order_2nd))
        # exclude 0th and 1st order coef
        coef2D = np.insert(np.insert(coef2D, 0, 0, axis=0), 0, 0, axis=1)
        # value
        return np.polynomial.legendre.legval(data_x_y[0], coef_t0) + np.polynomial.legendre.legval(data_x_y[1],
                                                                                                   coef_t1) + zero_t + \
            np.polynomial.legendre.legval2d(data_x_y[0], data_x_y[1], coef2D)

    def __init__(self, data_dir, data_sample, target_time, zero_order, my_coef1D, my_coef2D,
                 idx_pair_idx, index,  n_2_o_idx=None,
                 file_name="target_vs_K_1D_2D.jpg"):
        idx_2d = idx_pair_idx[tuple([index[0], index[1]])]
        self.Nth_order_2nd = int(np.sqrt(len(my_coef2D[idx_2d, :])))

        my_coef1D_1 = my_coef1D[index[0], :]
        my_coef1D_2 = my_coef1D[index[1], :]

        self.Nth_order_1st = len(my_coef1D_1)

        zero_first = [zero_order] + list(my_coef1D_1) + list(my_coef1D_2)
        zero_first_second = [zero_order] + \
            list(my_coef1D_1) + list(my_coef1D_2) + list(my_coef2D[idx_2d, :])

        x_1 = np.linspace(self.xmin, self.xmax)
        y_1 = np.linspace(self.xmin, self.xmax)
        x_g, y_g = np.meshgrid(x_1, y_1)

        fig, a_x = plt.subplots()
        #         plot1= ax.imshow(self.my_legendre_polynomial_2D_val([xg, yg], *my_coef2D), extent=[-1,1,1,-1], origin="upper")
        plot1 = a_x.imshow(self.func_leg_1D(
            [x_g, y_g], *zero_first), extent=[-1, 1, 1, -1], origin="upper")
        #         plot1= ax.imshow(self.my_legendre_polynomial_1D_2D_val([xg, yg], *zero_first_second), extent=[-1,1,1,-1], origin="upper")

        cb = fig.colorbar(plot1, ax=a_x)
        cb.formatter.set_scientific(True)
        cb.formatter.set_powerlimits((-2, 2))
        cb.update_ticks()

        label0 = str(index[0])
        label1 = str(index[1])
        if n_2_o_idx is not None:
            label0 = str(n_2_o_idx[int(label0)])
            label1 = str(n_2_o_idx[int(label1)])

        a_x.set_xlabel("$k_{" + label0 + "}$")
        a_x.set_ylabel("$k_{" + label1 + "}$")
        a_x.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        a_x.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

        file_name1 = file_name.split(".")[0] + "_" + label0 + "_" + label1 + "_contour." + \
            file_name.split(".")[-1]
        fig.savefig(os.path.join(data_dir, "output", file_name1), dpi=600)

        fig2 = plt.figure()
        a_x_2 = fig2.add_subplot(111, projection='3d')

        a_x_2.scatter(data_sample[:, 0], data_sample[:, 1], target_time,
                      c='#d3ffce', marker='o', s=1.0, alpha=0.8)

        a_x_2.plot_surface(x_g, y_g, self.my_legendre_polynomial_2D_val([x_g, y_g], *(my_coef2D[idx_2d, :])),
                           rstride=1, cstride=1,
                           cmap=plt.cm.get_cmap('jet'), linewidth=0, antialiased=False, alpha=0.8)
        a_x_2.plot_surface(x_g, y_g, self.func_leg_1D([x_g, y_g], *zero_first),
                           rstride=1, cstride=1,
                           cmap=plt.cm.get_cmap('jet'), linewidth=0, alpha=0.8)
        a_x_2.plot_surface(x_g, y_g, self.my_legendre_polynomial_1D_2D_val([x_g, y_g], *zero_first_second),
                           rstride=1,
                           cstride=1,
                           cmap=plt.cm.get_cmap('jet'), linewidth=0, alpha=0.8)

        a_x_2.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        a_x_2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        a_x_2.ticklabel_format(axis='z', style='sci', scilimits=(-2, 2))

        a_x_2.set_xlabel("$k_{" + label0 + "}$")
        a_x_2.set_ylabel("$k_{" + label1 + "}$")
        a_x_2.set_zlabel("target")
        a_x_2.set_title("$0^{th}$ + $1^{st}$ + $2^{nd}$ order fit function")
        #         ax2.view_init(37.5,-30)
        a_x_2.view_init(10.5, -15)

        #         ax2.view_init(20,70)
        #         ax2.view_init(100,-30)

        fig2.subplots_adjust(left=0.01, right=0.90, top=0.95)
        file_name2 = file_name.split(".")[0] + "_" + label0 + "_" + label1 + "." + \
            file_name.split(".")[-1]
        fig2.savefig(os.path.join(data_dir, "output", file_name2), dpi=600)
        plt.close('all')


def plot_fit_functions(data_dir, s_a_s=None, n_2_o_idx=None):
    """
    plot selected fit functions
    """
    u_norm = mu.read_uncertainty(os.path.join(data_dir, "output", "uncertainties_const.csv"),
                                 os.path.join(data_dir, "output", "k_global.csv"))
    target_sample = mu.read_target(os.path.join(
        data_dir, "output", "ign_global.csv"))

    _, zero_order_coef, first_order_coef, second_order_coef = prc.parse_regression_coef_c.get_var_zero_first_second_coef(
        data_dir, s_a_s=s_a_s)
    for idx in range(int(s_a_s['N_variable'])):
        plot_1D_c(data_dir, u_norm, target_sample,
                  first_order_coef, zero_order_coef, idx,
                  n_2_o_idx=n_2_o_idx)
    idx_pair_idx = flsr.fit_1D_2D_all_c.get_idx_pair_idx(s_a_s['N_variable'])
    for i in range(int(s_a_s['N_variable'])):
        for j in range(i + 1, int(s_a_s['N_variable'])):
            plot_2D_c(data_dir, u_norm, target_sample,
                      second_order_coef, idx_pair_idx, [i, j],
                      n_2_o_idx=n_2_o_idx)
            plot_1D_2D_c(data_dir, u_norm, target_sample,
                         zero_order_coef, first_order_coef, second_order_coef,
                         idx_pair_idx, index=[i, j],
                         n_2_o_idx=n_2_o_idx)
