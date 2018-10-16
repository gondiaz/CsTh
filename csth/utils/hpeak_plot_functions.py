# Functions to plot peak-hits
#
# J.A Hernando, 10/10/18


import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.dates  as md
from mpl_toolkits.mplot3d import Axes3D

import collections       as collections
import pandas            as pd

from invisible_cities.core .core_functions import in_range
import krcal.utils.hst_extend_functions   as hst

from csth.utils.hpeak_functions import selection_sipms


def check_pkslices(ez):
    c = hst.Canvas(4, 2)
    hst.plot(ez.Z, ez.E0, marker='o', label='E0', canvas=c(1), xylabels=('z (mm)', 'energy (pes)'))
    hst.plot(ez.Z, ez.E , marker='o', label='E' , canvas=c(1))
    plt.legend()
    hst.plot(ez.Z, ez.Q0, marker='o', label='Q0', canvas=c(2), xylabels=('z (mm)', 'energy (pes)'))
    hst.plot(ez.Z, ez.Q , marker='o', label='Q' , canvas=c(2))
    plt.legend()
    hst.plot(ez.Z,    ez.E0, marker='o', label='E0'   , canvas=c(3), xylabels=('z (mm)', 'energy (pes)'))
    hst.plot(ez.Z, 12*ez.Q0, marker='o', label='12*Q0', canvas=c(3))
    plt.legend()
    hst.plot(ez.Z,    ez.E, marker='o', label='E'   , canvas=c(4), xylabels=('z (mm)', 'energy (pes)'))
    hst.plot(ez.Z, 12*ez.Q, marker='o', label='12*Q', canvas=c(4))
    plt.legend()
    hst.plot(ez.Z, ez.N           , marker='o', canvas=c(5), xylabels=('z (mm)', 'number of hits'))
    #plt.yscale('log')
    hst.plot(ez.Z, ez.Q0/(1.*ez.N), marker='o', canvas=c(6), xylabels=('z (mm)', 'average Q0 (pes) per hit'))
    hst.plot(ez.Z, ez.E/ez.E0, marker='o', canvas=c(7), xylabels=('z (mm)', 'E/E0'))
    hst.plot(ez.Z, ez.Q/ez.Q0, marker='o', canvas=c(8), xylabels=('z (mm)', 'Q/Q0'))

    plt.tight_layout()
    return

def check_pkhits(hits):
    c = hst.Canvas(4, 2)
    sel = hits.Q0>0
    hst.scatter(hits.Z[sel], hits.Q0[sel], canvas=c(1), alpha=0.5, xylabels=('z (mm)', 'Q0 (pes)'))
    hst.scatter(hits.Z[sel], hits.F[sel] , canvas=c(2), alpha=0.5, xylabels=('z (mm)', 'fraction of charge'))
    hst.scatter(hits.Z[sel], hits.Q[sel], canvas=c(3), alpha=0.5, xylabels=('z (mm)', 'Q  (pes)'))
    hst.scatter(hits.Z[sel], hits.Q[sel]/hits.Q0[sel], canvas=c(4), alpha=0.5, xylabels=('z (mm)', 'Q/Q0'))
    hst.scatter(hits.Z[sel], hits.E[sel], canvas=c(5), alpha=0.5, xylabels=('z (mm)', 'E  (pes)'))
    hst.scatter(hits.Z[sel], hits.E[sel]/hits.E0[sel], canvas=c(6), alpha=0.5, xylabels=('z (mm)', 'E/E0'))
    plt.tight_layout()
    return

from mpl_toolkits.mplot3d import Axes3D

def graph_3D(hits):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax3D = fig.add_subplot(111, projection='3d')
    sel = hits.E>0
    emax = np.max(hits.E[sel])
    x, y, z, col = hits.X[sel], hits.Y[sel], hits.Z[sel], hits.E[sel]
    p3d = ax3D.scatter(z, x, y, s=30, c=col, alpha=0.4, marker='o')
    return

def graph_xy(hits):
    nzs    = len(set(hits.K))
    nsipms = int(len(hits.K)/nzs)
    sels = selection_sipms(nzs, nsipms)
    x   = [hits.X[sel].values[0] for sel in sels]
    y   = [hits.Y[sel].values[0] for sel in sels]
    ene = [np.sum(hits.Q[sel])   for sel in sels]
    hst.scatter(x, y, c=ene, cmap='jet');
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    title = 'event '+str(*set(hits.event))+', S2 '+str(*set(hits.peak))
    plt.title(title)
    plt.colorbar();
    return

def graph_hits(hits):
    c = hst.Canvas(2, 2)
    sel = hits.Q0>0
    x, y, z, ene = hits.X[sel], hits.Y[sel], hits.Z[sel], hits.E[sel]
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.gcf()
    ax3D = fig.add_subplot(221, projection='3d')
    p3d = ax3D.scatter(z, x, y, s=20, c=ene, alpha=0.4, marker='o')
    hst.scatter(x, z, c=ene, alpha=0.4, canvas=c(2), cmap='jet', xylabels=('x (mm)', 'z (mm)'))
    plt.colorbar();
    hst.scatter(z, y, c=ene, alpha=0.4, canvas=c(3), cmap='jet', xylabels=('z (mm)', 'y (mm)'))
    plt.colorbar();
    hst.scatter(x, y, c=ene, alpha=0.4, canvas=c(4), cmap='jet', xylabels=('x (mm)', 'y (mm)'))
    plt.colorbar();
    plt.tight_layout()
    return


def check_pksum(pk):
    c = hst.Canvas(6, 2)

    hst.hist(pk.nslices, 100, canvas=c(1) , xylabels=('nslices', ''))
    hst.hist(pk.nsipms , 100, canvas=c(2) , xylabels=('nsipms', ''))
    hst.hist(pk.N      , 100, canvas=c(3) , xylabels=('hits/slice', ''))
    hst.hist(pk.Z      , 100, canvas=c(4) , xylabels=('X', ''))
    hst.hist(pk.X      , 100, canvas=c(5) , xylabels=('Y', ''))
    hst.hist(pk.Y      , 100, canvas=c(6) , xylabels=('Z', ''))

    hst.hist(pk.Q0     , 100, canvas=c(7) , xylabels=('Q0', ''))
    hst.hist(pk.Q      , 100, canvas=c(8) , xylabels=('Q', ''))
    hst.hist(pk.E0     , 100, canvas=c(9) , xylabels=('E0', ''))
    hst.hist(pk.E      , 100, canvas=c(10), xylabels=('E', ''))

    hst.hist2d(pk.X, pk.Y, (40, 40), canvas=c(11), xylabels=('X', 'Y'))
    hst.hist2d(pk.Z, pk.E, (40, 40), canvas=c(12), xylabels=('E', 'Z'))
    plt.tight_layout()
    return
