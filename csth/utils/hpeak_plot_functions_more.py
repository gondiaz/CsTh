

import matplotlib.pyplot as plt
import matplotlib.dates  as md

import numpy as np

import krcal.utils.hst_extend_functions   as hst


def table_plot_var(tab, names, nbins=100):

    n = int(len(names)/2)+1
    c = hst.Canvas(n, 2)
    for i, name in enumerate(names):
        hst.hist(getattr(tab, name), nbins, canvas = c(i+1), xylabels=(name, ''))
    return

def table_plot_ratio(tab, pair_names, nbins=100):
    n = len(pair_names)
    c = hst.Canvas(n, 2)
    for i, pair_name in enumerate(pair_names):
        n1, n2 = pair_name
        v1, v2 = getattr(tab, n1), getattr(tab, n2)
        c(2*i+1)
        plt.scatter(v1, v2)
        tsel = abs(v1) >0
        hst.hist(v2[tsel]/v1[tsel], nbins, canvas = c(2*i+2), xylabels=(n2+'/'+n1, ''))
    return


def table_plot_event_slices(stab, evt, pk):
    ssel = np.logical_and(stab.event == evt, stab.peak == pk)

    c = hst.Canvas(3, 2)
    hst.plot(stab.z0[ssel], stab.e0[ssel], marker='o', label='E0', canvas = c(1))
    hst.plot(stab.z0[ssel], stab.e [ssel], marker='o', label='E ', canvas = c(1))
    plt.legend()

    hst.plot(stab.z0[ssel], stab.q0[ssel], marker='o', label='Q0', canvas = c(2))
    hst.plot(stab.z0[ssel], stab.q [ssel], marker='o', label='Q ', canvas = c(2))
    plt.legend()

    hst.plot(stab.z0[ssel], stab.x0[ssel], marker='o',  label='X0', canvas = c(3))
    hst.plot(stab.z0[ssel], stab.x [ssel], marker='o', label='X' , canvas = c(3))
    plt.legend()

    hst.plot(stab.z0[ssel], stab.y0[ssel], marker='o', label='Y0', canvas = c(4))
    hst.plot(stab.z0[ssel], stab.y [ssel], marker='o', label='Y' , canvas = c(4))
    plt.legend()

    hst.plot(stab.z0[ssel], stab.nhits [ssel], marker='o', canvas = c(5), xylabels=('nhits',''))

    plt.legend()


def table_plot_event_hits(stab, evt, pk = 0):
    ssel = np.logical_and(stab.event == evt, stab.peak == pk)

    c = hst.Canvas(2, 2)
    c(1)
    plt.scatter(stab.z0[ssel], stab.e0[ssel], marker='o', alpha=0.5, label='E0')
    plt.scatter(stab.z0[ssel], stab.e [ssel], marker='o', alpha=0.5, label='E ')
    plt.legend()

    c(2)
    plt.scatter(stab.z0[ssel], stab.q0[ssel], marker='o', alpha=0.5, label='Q0')
    plt.scatter(stab.z0[ssel], stab.q [ssel], marker='o', alpha=0.5, label='Q ')
    plt.legend()

    c(3)
    plt.scatter(stab.z0[ssel], stab.x0[ssel], marker='o', c = stab.e[ssel], label='X0')
    #plt.ylim(-200., 200.)

    c(4)
    plt.scatter(stab.z0[ssel], stab.y0[ssel], marker='o', c = stab.e[ssel], label='Y0')
    #plt.ylim(-200., 200.)
    return
