import os
import numpy             as np
import collections       as collections
import pandas            as pd

#---------------------------
#    Tables definitions
#---------------------------

#EventList     = collections.namedtuple('EventList', ['event', 'peak'])

#EventFullList = collections.namedtuple('EventFullList', ['event', 'peak', 'nslices', 'nhits'])

event_nints = 8
event_names = ['event', 'peak', 'location', 'nslices', 'nhits', 'noqslices', 'noqhits', 'time',
               's1e', 't0', 'rmax', 'rbase', 'zmin', 'zmax',
               'x0', 'y0', 'z0', 'q0', 'e0',
               'x', 'y', 'z', 'q', 'e', 'qc', 'ec']

slice_nints = 2
slice_names = ['event', 'peak', 'z0', 'q0', 'e0', 'q' , 'e' ]

hit_nints   = 2
hit_names   = ['event', 'peak', 'x0', 'y0', 'z0', 'q0', 'e0', 'q', 'e']

class Table:

    def __init__(self, names, nints = 0):
        self.nvars = len(names)
        self.names = names
        self.nints = nints

    def size(self):
        name0 = self.names[0]
        val = getattr(self, name0)
        if (type(val) != np.ndarray): return 1
        return len(val)

    def zeros(self, size):
        for i in range(self.nints):
            dat = 0  if size <=1 else np.zeros(size, dtype = int)
            setattr(self, self.names[i], dat )
        for i in range(self.nints, self.nvars):
            dat = 0. if size <=1 else np.zeros(size, dtype = float)
            setattr(self, self.names[i], dat )
        return

    def __str__(self):
        ss = ''
        for name in self.names:
            ss += ' ' + name  + ': ' + str(getattr(self, name))+', '
        return ss


def event_table(size):
    etab = Table(event_names, event_nints)
    etab.zeros(size)
    return etab


def slice_table(size):
    stab = Table(slice_names, slice_nints)
    stab.zeros(size)
    return stab


def hit_table(size):
    htab = Table(hit_names, hit_nints)
    htab.zeros(size)
    return htab


def set_table(table_origen, eindex, table):
    size = table_origen.size()
    for name in table.names:
        getattr(table, name) [eindex: eindex + size] = getattr(table_origen, name)
    return eindex + size


def df_from_table(table):
    df = {}
    for name in table.names:
        df[name] = getattr(table, name)
    return pd.DataFrame(df)


#-----------------------------
# Generic hits code
#-----------------------------

def event_eqpoint(e0i, z0i, x0ij, y0ij, q0ij):
    """ compute the average point position and total charge and energy of
    an event-peak
    inputs:
        e0i      : (array) raw energy per slice
        z0i      : (array) z position of the slices
        x0ij     : (array) x position of the hits
        y0ij     : (array) y position of the hits
        z0ij     : (array) z position of the hits
        q0ij     : (array) raw charge of the hits
    returns:
        x, y, z  : (float, float, float) average position
        q, e     : (float, float)        total charge and energy

    """

    e0 = np.sum(e0i)
    if (e0 <= 1.): e0 = 1.
    z0    = np.sum(z0i*e0i)/e0

    q0 = np.sum(q0ij)
    if (q0 <= 1.): q0 = 1.
    x0    = np.sum(x0ij*q0ij)/q0
    y0    = np.sum(y0ij*q0ij)/q0

    #print('x, y, z, q, e ', x0, y0, z0, q0, e0)
    return x0, y0, z0, q0, e0


def hits_energy(e0i, z0i, z0ij, q0ij, ceij = None, cqij = None):
    """ share energy between hits and correct it by a factor
    inputs:
        e0i      : (array) energy per slice
        z0i      : (array) z position of the slices
        z0ij     : (array) z position of the hits
        q0ij     : (array) charge of the hits
        ceij     : (array) energy correction factor per hit (default 1.)
        cqij     : (array) cahrge correction factor per hit (default 1.)
    returns:
        ei       : (array) corrected energy per slice
        qi       : (array) charge por slice
        eij      : (array) energy per hit
        qij      : (array) energy per hit
    """
    nslices = len(z0i)
    nhits   = len(z0ij)

    qij = q0ij * cqij if cqij is not None else q0ij * 1.

    selslices = selection_slices_by_z(z0ij, z0i)

    qi  = np.array([np.sum(qij[sel]) for sel in selslices])
    eij = np.ones(nhits)
    qi [qi <= 1.] = 1.
    for k, kslice in enumerate(selslices):
        eij[kslice] = qij[kslice] * e0i[k]/qi[k]

    if (ceij is not None): eij = eij * ceij

    ei = np.array([np.sum(eij[sel]) for sel in selslices])

    return ei, qi, eij, qij

def slices_energy(e0i, ei):
    """ correct energy per slices without hits
    inputs:
        e0i     : (array) raw energy per slice
        ei      : (array) corrected energy per slice
    returns:
        ei      : (array) corrected energy per slice (including slices with no hits)
        enoq    : (float) corrected energy in the slices with no hits
        nqslices: (int)   number of slices without hits
    """

    e0i [e0i <= 1] = 1.
    fi             = ei/e0i
    selnoq         = fi <= 0.
    noqslices      = np.sum(selnoq)
    if (noqslices > 0):
        fmed           = np.mean(fi[~selnoq])
        ei [selnoq]    = fmed * e0i [selnoq]
    enoq = np.sum(ei[selnoq])

    return ei, enoq, noqslices

def calibration_factors(x, y, z, calibrate):

    nhits = len(z)
    ones = np.ones(nhits)

    #ce0, cq0 = calibrate(x, y, None, None, ones, ones)
    ce , cq  = calibrate(x, y, z   , None, ones, ones)

    #ce0 [ce0 <= 0.] = 1.
    #fe, fq  = ce, cq0*ce/ce0
    fe, fq  = ce, cq

    return fe, fq

"""
def calibrate_hits(e0i, z0i, x0ij, y0ij, z0ij, q0ij, calibrate):
    compute calibrated hits
    inputs:
        e0i      : (array) raw energy per slice
        z0i      : (array) z position of the slices
        x0ij     : (array) x position of the hits
        y0ij     : (array) y position of the hits
        z0ij     : (array) z position of the hits
        q0ij     : (array) raw charge of the hits
        calibrate: (function) calibrate function
        calq     : (bool)  calibrate or not the charge (default = True)
    returns:
        noqslices: (int)   number of slices without charge
        ei       : (array) corrected energy per slice
        qi       : (array) corrected charge por slice
        eij      : (array) corrected energy per slice and hit
        qij      : (array) corrected charge per slice and hit


    nslices = len(z0i)
    nhits = len(z0ij)
    eones = np.ones(nhits)
    fij, qij   = calibrate(x0ij, y0ij, z0ij, None, eones, q0ij)
    if (not calq):
        qij = q0ij

    selslices = selection_slices_by_z(z0ij, z0i)

    qi  = np.array([np.sum(qij[sel]) for sel in selslices])
    eij = np.ones(nhits)
    qi [qi <= 1.] = 1.
    for k, kslice in enumerate(selslices):
        eij [kslice]     = fij[kslice] * qij [kslice]*e0i[k]/qi [k]

    ei = np.array([np.sum(eij[sel]) for sel in selslices])

    e0i [e0i <= 1] = 1.
    fi             = ei/e0i
    selnoq         = fi <= 0.
    noqslices      = np.sum(selnoq)
    if (noqslices > 0):
        fmed           = np.mean(fi[~selnoq])
        ei [selnoq]    = fmed * e0i [selnoq]

    #print('ei ', len(ei), np.sum(ei), ei)
    #print('qi ', len(qi), np.sum(qi), qi)
    #print('eij ', len(eij), eij)
    #print('qij ', len(qij), qij)

    return noqslices, ei, qi, eij, qij
"""

def radius(x0ij, y0ij, x0, y0):
    """ returns the maximum radius respect the origin and (x0, y0) or base radius
    inputs:
        x0ij  : (array) x position of the hits
        y0ij  : (array) y position of the hits
        x0    : (float) x center
        y0    : (float) y center
    returns:
        rmax  : (float) the maximum radius of the hits (repect origin)
        rbase : (float) the maximun radius respect (x0, y0) or base radius
    """
    def _rad(x, y):
        r2 = x*x + y*y
        rmax = np.sqrt(np.max(r2))
        return rmax
    rmax  = _rad( x0ij     , y0ij      )
    rbase = _rad( x0ij - x0, y0ij - y0 )

    #print('max radius, base radius ', rmax, rbase)
    return rmax, rbase


def zrange(z0i):
    """ return the range of the z values minimum and maximum
    inputs:
        z0i       :  (array) z positions of the slices
    returns:
        zmin, zmax:  (float, float) z min and max
    """
    return np.min(z0i), np.max(z0i)

#--------------------------------------------
#   Utilities for selection of slices or sipms
#--------------------------------------------

"""
def selection_slices(nzs, nsipms):
    def _slice(i):
        sx = np.array(nsipms*nzs*[False])
        for k in range(nsipms):
            sx[k*nzs+i] = True
        return sx
    return [_slice(i) for i in range(nzs)]

def selection_sipms(nzs, nsipms):
    def _slice(i):
        sx = np.array(nsipms*nzs*[False])
        for k in range(nzs):
            sx[i*nzs+k] = True
        return sx
    return [_slice(i) for i in range(nsipms)]

"""

def selection_slices_by_z(zij, zi):
    sels = [zij == izi for izi in zi]
    return sels

def selection_slices_by_slice(islices, nslices):
    return [islices == i for i in range(nslices)]


#-----------------------------------
#
#-----------------------------------
