import matplotlib.pyplot as plt
import matplotlib.dates  as md
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

import krcal.utils.hst_extend_functions   as hst


def plot_var(tab, names, nbins=100):

    n = int(len(names)/2)+1
    c = hst.Canvas(n, 2)
    for i, name in enumerate(names):
        hst.hist(getattr(tab, name), nbins, canvas = c(i+1), xylabels=(name, ''))
    return

def plot_ratio(tab, pair_names, nbins=100):
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


def plot_vs_z(hits, q0min = 0.):
    c = hst.Canvas(5, 2)
    sel = hits.q0 > q0min
    hst.scatter(hits.z0[sel], hits.q0[sel], canvas=c(1), alpha=0.5, xylabels=('z (mm)', 'q0 (pes)'))
    hst.scatter(hits.z0[sel], hits.q [sel], canvas=c(2), alpha=0.5, xylabels=('z (mm)', 'q (pes)'))
    hst.scatter(hits.z0[sel], hits.e0[sel], canvas=c(3), alpha=0.5, xylabels=('z (mm)', 'e0  (pes)'))
    hst.scatter(hits.z0[sel], hits.e[sel] , canvas=c(4), alpha=0.5, xylabels=('z (mm)', 'e '))
    hst.scatter(hits.z0[sel], hits.x0[sel], canvas=c(5), alpha=0.2,
                c=hits.e[sel], s= 0.1*hits.e[sel], xylabels=('z (mm)', 'x  (mm)'))
    hst.scatter(hits.z0[sel], hits.y0[sel], canvas=c(6), alpha=0.1,
                c=hits.e[sel], s= 0.1*hits.e[sel], xylabels=('z (mm)', 'y (mm)'))
    hst.scatter(hits.z0[sel], hits.q[sel]/hits.q0[sel], canvas=c(7), alpha=0.5, xylabels=('z (mm)', 'q/q0'))
    hst.scatter(hits.z0[sel], hits.e[sel]/hits.e0[sel] , canvas=c(8), alpha=0.5, xylabels=('z (mm)', 'e/e0'))
    plt.tight_layout()
    return

def graph_hits(hits, q0min = 0):
    c = hst.Canvas(2, 2)
    sel = hits.q0 > q0min
    x, y, z, ene = hits.x0[sel], hits.y0[sel], hits.z0[sel], hits.e[sel]
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.gcf()
    ax3D = fig.add_subplot(221, projection='3d')
    p3d = ax3D.scatter(z, x, y, s=0.1*ene, c=ene, alpha=0.4, marker='o')
    ax3D.set_xlabel('z (mm)')
    ax3D.set_ylabel('x (mm)')
    ax3D.set_zlabel('y (mm)')
    hst.scatter(x, z, c=ene, s=0.1*ene, alpha=0.2, canvas=c(2), cmap='jet', xylabels=('x (mm)', 'z (mm)'))
    plt.colorbar();
    hst.scatter(z, y, c=ene, s=0.1*ene, alpha=0.2, canvas=c(3), cmap='jet', xylabels=('z (mm)', 'y (mm)'))
    plt.colorbar();
    hst.scatter(x, y, c=ene, s=0.1*ene, alpha=0.2, canvas=c(4), cmap='jet', xylabels=('x (mm)', 'y (mm)'))
    plt.colorbar();
    plt.tight_layout()
    return

def graph_event(x, y, z, ene, scale = 0.1, comment = ''):
    c = hst.Canvas(2, 2)
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.gcf()
    ax3D = fig.add_subplot(221, projection='3d')
    p3d = ax3D.scatter(z, x, y, s=scale*ene, c=ene, alpha=0.4, marker='o')
    ax3D.set_xlabel('z (mm)')
    ax3D.set_ylabel('x (mm)')
    ax3D.set_zlabel('y (mm)')
    plt.title(comment)
    hst.scatter(x, z, c=ene, s=scale*ene, alpha=0.2, canvas=c(2), cmap='jet', xylabels=('x (mm)', 'z (mm)'))
    plt.colorbar();
    hst.scatter(z, y, c=ene, s=scale*ene, alpha=0.2, canvas=c(3), cmap='jet', xylabels=('z (mm)', 'y (mm)'))
    plt.colorbar();
    hst.scatter(x, y, c=ene, s=scale*ene, alpha=0.2, canvas=c(4), cmap='jet', xylabels=('x (mm)', 'y (mm)'))
    plt.colorbar();
    plt.tight_layout()
    return
