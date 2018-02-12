"""
tools, for example routine helping making figures
"""
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
from matplotlib.lines import Line2D

import numpy as np

def get_colors_markers_linestyles():
    """
    return colors
    markers
    linestyles
    """
    markers_tmp = []
    for m_k in Line2D.markers:
        try:
            if len(m_k) == 1 and m_k != ' ':
                markers_tmp.append(m_k)
        except TypeError:
            pass

    markers_tmp = markers_tmp + [
        r'$\lambda$',
        r'$\bowtie$',
        r'$\circlearrowleft$',
        r'$\clubsuit$',
        r'$\checkmark$']

    markers = markers_tmp[2::]
    markers.append(markers_tmp[0])
    markers.append(markers_tmp[1])

    colors = ('b', 'g', 'k', 'c', 'm', 'y', 'r')
    linestyles = Line2D.lineStyles.keys()

    return colors, markers, linestyles


def make_figure_template(data_dir):
    """
    make_figure_template
    """
    colors, markers, _ = get_colors_markers_linestyles()
    # x axis file name
    f_n_x = "fname_x.csv"
    # y axis file name
    f_n_y = "fname_y.csv"
    # figure name
    fig_name = "test.jpg"

    data_x = np.loadtxt(os.path.join(data_dir, f_n_x),
                        dtype=float, delimiter=",")
    data_y = np.loadtxt(os.path.join(data_dir, f_n_y),
                        dtype=float, delimiter=",")
    # specify label for lines
    labels = ["line" + str(i + 1) for i in range(len(data_y))]

    delta_n = int(len(data_x) / 25)
    if delta_n is 0:
        delta_n = 1

    fig, a_x = plt.subplots(1, 1, sharex=True, sharey=False)
    for idx, _ in enumerate(data_y):
        a_x.plot(data_x[::delta_n], data_y[idx, ::delta_n],
                 color=colors[idx], marker=markers[idx], label=labels[idx])

    leg = a_x.legend(loc=0, fancybox=True, prop={'size': 10.0})
    leg.get_frame().set_alpha(0.7)

    a_x.set_xlim([data_x[0], data_x[-1]])
    a_x.grid()

    a_x.set_xlabel("1000/T(K$^{-1}$)")
    a_x.set_ylabel("k(cm$^{3}$ molecule$^{-1}$s$^{-1}$)")
    a_x.set_title("O$_2$ + npropyl")

    fig.tight_layout()
    fig.savefig(os.path.join(data_dir, fig_name), dpi=500)
    plt.close()