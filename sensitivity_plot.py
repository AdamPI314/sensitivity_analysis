"""
sensitivity plot
"""
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
from matplotlib.ticker import FormatStrFormatter

import global_settings
from color_marker import get_colors_markers_linestyles


def bar_1D_SI(file_dir, n_2_o_idx=None):
    """
    bar
    """

    colors, markers, _ = get_colors_markers_linestyles()

    SI_1st_data = np.loadtxt(os.path.join(
        file_dir, "output", "SI_1st.csv"), delimiter=',', dtype=float)

    # figure object
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False)

    n_size = len(SI_1st_data)
    x = np.arange(n_size)    # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    bar1 = ax.bar(x, SI_1st_data, width, color=colors,
                  bottom=0, align="center")

    ax.set_xticks(x)
    if n_2_o_idx is None:
        ticks = [str(i) for i in range(n_size)]
    else:
        ticks = [str(n_2_o_idx[i]) for i in range(n_size)]

    ax.set_xticklabels(ticks, rotation=45, fontsize=8)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    ax.margins(0.01)
    # ax.set_xbound(0, 23)
    ax.set_ybound(0, 1.0)

    ax.set_xlabel("species")
    ax.set_ylabel("$1^{st}$" + " order SI")
    ax.set_title("$1^{st}$" + " order SI" +
                 " of ignition delay time wrt. initial concentration")
    ax.text(1, 1.0 - 0.1, "SUM($1^{st}$ order SI): " +
            "%1.4g" % (np.sum(SI_1st_data)))

    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=None, hspace=None)

    o_f_n = os.path.join(file_dir, "output", "SI_1st.png")
    fig.savefig(o_f_n, dpi=500, bbox_inches='tight')
    plt.close(fig)
