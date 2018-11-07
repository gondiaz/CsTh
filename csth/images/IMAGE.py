import matplotlib as mpl
#mpl.use('Qt5Agg')
mpl.use('nbAgg')

import numpy               as np
import pandas              as pd
import matplotlib.pyplot   as plt
import matplotlib.patches  as patches
import matplotlib.gridspec as gridspec

from mpl_toolkits.mplot3d    import Axes3D
from matplotlib.path         import Path
from matplotlib.animation    import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets      import Slider


def _each(edges, n):
    if len(edges)%n:
        mat = []
        mit = []
        i = 0
        while i<len(edges)-n:
            mat.append(edges[i])
            for j in range(1,n): mit.append(edges[i+j])
            i += n
        mat.append(edges[-1])
        return mat, mit
    else: raise ValueError("number of edges do not match with n")


def _NEW_TP_edge():
    verts = [(-80 , -240), (80  , -240), (80  , -200), (160 , -200), (160 , -120),
             (240 , -120), (240 , 120) , (160 , 120) , (160 , 200) , (80  , 200) ,
             (80  , 240) , (-80 , 240) , (-80 , 200) , (-160, 200) , (-160, 120) ,
             (-240, 120) , (-240, -120), (-160, -120), (-160, -200), (-80 , -200),
             (-80 , -240)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO,
             Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO,
             Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO,
             Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO,
             Path.CLOSEPOLY,]
    path  = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor='black', lw=3)
    return patch


class IMAGE:

    def __init__(self, imagedf, datasipm, run, event, typo):
        self.im  = imagedf
        self.z   = np.unique(imagedf['Z'])
        self.dsi = datasipm
        self.r   = run
        self.ev  = event
        self.ty  = typo

        self.fig     = 0
        self.si_enes = []
        self.bins    = 0
        self.rg      = 0
        self.edges   = 0
        self.axim    = 0
        self.ez      = 0
        self.e       = 0
        self.t       = 0
        self.cir     = 0
        self.sl      = 0


    def _initvalues(self):
        d = 10 #distance between Sipms

        x   , y    =  self.dsi['X'], self.dsi['Y']
        xmin, xmax =  x.min()      , x.max()
        ymin, ymax =  y.min()      , y.max()

        nx  , ny   = (xmax-xmin)/10 + 1   , (ymax-ymin)/10 + 1
        rg         = [[xmin-d/2, xmax+d/2], [ymin-d/2, ymax+d/2]]

        H, xedges, yedges = np.histogram2d(x, y, bins = [nx, ny], range = rg)
        H = H.T

        for z in self.z:
            self.si_enes.append(self.im[self.im.Z == z]['E'].sum())

        self.bins  = [nx, ny]
        self.rg    = rg
        self.edges = [xedges, yedges]
        return H


    def init_figure(self):
        self.fig     =  plt.figure(constrained_layout=True)

        spec = gridspec.GridSpec(ncols=2, nrows=1, left=0.05, right=0.95, wspace=0.25)

        ax  = self.fig.add_subplot(spec[0, 0], title='NEW tracking plane')
        ax1 = self.fig.add_subplot(spec[0, 1], title='E vs Z')

        H = self._initvalues()

        #TRACKING PLANE
        enemin, enemax  = self.im['E'].min(), self.im['E'].max()
        xedges, yedges  = self.edges[0]       , self.edges[1]
        self.axim = ax.imshow(H, cmap='jet', vmin = enemin, vmax = enemax, interpolation='nearest',
                         origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], alpha = 0.5,
                         aspect='auto')

        #ticks of 1 cm^2
        maxt, mixt = _each(xedges, n = 4)
        ax.set_xticks(maxt, minor = False)
        ax.set_xticks(mixt, minor = True)

        mayt, miyt = _each(yedges, n = 4)
        ax.set_yticks(mayt, minor = False)
        ax.set_yticks(miyt, minor = True)
        #colorbar
        divider = make_axes_locatable(ax)
        cax     = divider.append_axes("right", size="3%", pad=0.07)
        cbar    = self.fig.colorbar(self.axim, cax = cax);

        # E vs T AXES
        self.ez, = ax1.plot(self.z, self.si_enes)


    def _figure_customize(self):
        self.fig.text    ( .1, .95 , f'Run: {self.r} \t Event:{self.ev} \t Type: {self.ty}');
        #self.fig.text    ( .1, .9  , 'Event number: {}'.format(self.ev));
        #self.fig.text    ( .1, .85 , 'Event type: {}'  .format(self.ty));
        e = self.fig.text( .56, .85 , 'Energy: {}'      .format(''));
        t = self.fig.text( .56, .8 ,  'Z: {}'        .format(''));
        self.e = e
        self.t = t


    def _TP_customize(self):
        # grid, detector contourn
        ax = self.fig.axes[0]
        ax.grid(which='both');
        patch = _NEW_TP_edge()
        ax.patch.set_animated(True);
        ax.add_patch(patch);


    def _ET_customize(self, xy):
        #draw circle (taken from https://werthmuller.org/blog/2014/circle/)
        # circle centre
        radius = .01
        ax1 = self.fig.axes[1]

        pr      = self.fig.get_figwidth()/self.fig.get_figheight()
        tscale  = ax1.transScale + (ax1.transLimits + ax1.transAxes)
        ctscale = tscale.transform_point(xy)
        cfig    = self.fig.transFigure.inverted().transform(ctscale)

        cir = patches.Ellipse(cfig, radius, radius*pr,
        transform=self.fig.transFigure, facecolor = 'none', edgecolor = 'r')

        ax1.add_artist(cir);

        self.cir = cir


    def update(self, frame_number):
        '''Update function for plots'''
        frame_number = int(frame_number)
        #sl.set_val(frame_number)
        #TRACKING PLANE
        imdf = self.im[self.im.Z == self.z[frame_number]]
        x, y, w = imdf['X'], imdf['Y'], imdf['E']
        H, __, __ = np.histogram2d(x, y, weights = w, bins = self.edges, range = self.rg)
        H = H.T
        self.axim.set_data(H)

        #ENERGY VS TIME plot
        ax1 = self.fig.axes[1]
        xy = (self.z[frame_number], w.sum())
        tscale   = ax1.transScale + (ax1.transLimits + ax1.transAxes)
        ctscale  = tscale.transform_point(xy)
        cfig     = self.fig.transFigure.inverted().transform(ctscale)
        self.cir.center = cfig

        # figure text actualization
        self.e.set_text('Energy: {}'.format(int(w.sum())))
        self.t.set_text('Z: {}'  .format(int(self.z[frame_number])))

        self.fig.canvas.draw_idle()


    def Movie_Slider(self):
        self.init_figure()
        self._figure_customize()
        self._TP_customize()
        self._ET_customize(xy = (self.z[0], self.si_enes[0]))


        #Slider plot creation
        caxsl = plt.axes([0.1, 0.01, 0.8, 0.03], facecolor=None)
        self.sl = Slider(caxsl, 'Slide', 0, len(self.z)-1, valinit = 0, valstep = 1)
        self.sl.on_changed(self.update);

    def plot3d(self, cut=0):
        from mpl_toolkits.mplot3d import Axes3D

        self.fig = plt.figure(constrained_layout=True)
        ax = self.fig.add_subplot(111, projection='3d')

        imdf = self.im[self.im.E > cut]
        X, Y, Z, E = imdf['X'], imdf['Y'], imdf['Z'], imdf['E']
        #X, Y, Z, E = self.im['X'], self.im['Y'], self.im['Z'], self.im['E']

        im3d = ax.scatter(Z, X, Y, zdir='z', s=20, c=E, depthshade=True, marker='s', alpha=0.5, cmap='jet');
        ax.set_xlabel('Z');
        ax.set_ylabel('X');
        ax.set_zlabel('Y');

        x   , y    =  self.dsi['X'], self.dsi['Y']
        xmin, xmax =  x.min()      , x.max()
        ymin, ymax =  y.min()      , y.max()
        ax.set_ylim([xmin, xmax]);
        ax.set_zlim([ymin, ymax]);

        cbar    = self.fig.colorbar(im3d, ax=ax, fraction=0.03, shrink=0.7);
