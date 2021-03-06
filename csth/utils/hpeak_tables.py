import os
import numpy             as np
import collections       as collections
import pandas            as pd

#---------------------------
#    Tables definitions
#---------------------------

EventList = collections.namedtuple('EventList', ['event', 'peak'])

etable_names = ['event', 'peak', 'loc', 'nslices', 'nhits', 'noqslices', 'noqhits',
                'sid', 'hid', 'time',
                's1e', 't0', 'rmax', 'zmin', 'zmax',
                'x0', 'y0', 'z0', 'q0', 'e0',
                'x' , 'y' , 'z' , 'q' , 'e' ]

edf_names    = ['event', 'peak', 'loc', 'nslices', 'nhits', 'noqslices', 'noqhits',
                'time', 's1e', 't0', 'rmax', 'zmin', 'zmax',
                'x0', 'y0', 'z0', 'q0', 'e0',
                'x', 'y', 'z', 'q', 'e']

ETable = collections.namedtuple('ETable', etable_names)

ETuple = collections.namedtuple('ETuple', edf_names)

stable_names = ['event', 'peak', 'slice', 'nhits',
                'rmax',
                'x0', 'y0', 'z0', 'q0', 'e0',
                'x' , 'y'       , 'q' , 'e' ]

STable = collections.namedtuple('STable', stable_names)

htable_names = ['event', 'peak', 'slice', 'nsipm',
                'x0', 'y0', 'z0', 'q0', 'e0', 'q', 'e']


HTable = collections.namedtuple('HTable', htable_names)

#------------------------------------
#   Utilities for tables
#------------------------------------

def _table(size, nint, ntot):
    items = [np.zeros(size, dtype = int) for i in range(nint)]
    items += [np.zeros(size) for i in range(nint, ntot)]
    return items

def create_event_table(size):
    return ETable(*_table(size, 10, len(etable_names)))

def create_slice_table(size):
    return STable(*_table(size, 4, len(stable_names)))

def create_hit_table(size):
    return HTable(*_table(size, 4, len(htable_names)))

def df_from_table(tab, names):
    df = {}
    for name in names:
        df[name] = getattr(tab, name)
    return pd.DataFrame(df)

def df_from_etable(tab):
    return df_from_table(tab, edf_names)

def df_from_stable(tab):
    return df_from_table(tab, stable_names)

def df_from_htable  (tab):
    return df_from_table(tab, htable_names)

def _etable_set(etable, etup, eindex):

    for name in edf_names:
         getattr(etable, name) [eindex] = getattr(etup, name)
    eindex += 1
    return eindex
    #return eindex+=1

    etable.event [eindex] = etup.event
    etable.peak  [eindex] = etup.peak
    etable.loc   [eindex] = etup.loc

    etable.time [eindex] = etup.time
    etable.s1e  [eindex] = etup.s1e
    etable.t0   [eindex] = etup.t0

    etable.nslices [eindex] = etup.nslices
    etable.nhits   [eindex] = etup.nhits
    etable.noqhits [eindex] = etup.noqhits

    etable.rmax [eindex] = etup.rmax
    etable.zmin [eindex] = etup.zmin
    etable.zmax [eindex] = etup.zmax

    etable.x0 [eindex] = etup.x0
    etable.y0 [eindex] = etup.y0
    etable.z0 [eindex] = etup.z0
    etable.q0 [eindex] = etup.q0
    etable.e0 [eindex] = etup.e0

    etable.x [eindex] = etup.x
    etable.y [eindex] = etup.y
    etable.z [eindex] = etup.z
    etable.q [eindex] = etup.q
    etable.e [eindex] = etup.e

    eindex += 1
    return eindex

def event_eqpoint(e0i, z0i, x0ij, y0ij, q0ij):

    e0 = np.sum(e0i)
    if (e0 <= 1.): e0 = 1.
    z0    = np.sum(z0i*e0i)/e0

    q0 = np.sum(q0ij)
    if (q0 <= 1.): q0 = 1.
    x0    = np.sum(x0ij*q0ij)/q0
    y0    = np.sum(y0ij*q0ij)/q0

    #print('x, y, z, q, e ', x0, y0, z0, q0, e0)
    return x0, y0, z0, q0, e0

def calibrate_hits(e0i, z0i, x0ij, y0ij, z0ij, q0ij, calibrate):

    nslices = len(z0i)
    nhits = len(z0ij)
    eones = np.ones(nhits)
    fij, qij   = calibrate(x0ij, y0ij, z0ij, None, eones, q0ij)

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

    return noqslices, ei, eij, qij


def max_radius_hit(x0ij, y0ij):
    r2 = x0ij*x0ij + y0ij*y0ij
    rmax = np.max(r2)
    rmax = np.sqrt(rmax)
    #print('max radius ', rmax)
    return rmax

def zrange(z0i):
    return np.min(z0i), np.max(z0i)

#--------------------------------------------
#   Utilities for selection of slices or sipms
#--------------------------------------------

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

def selection_slices_by_z(zij, zi):
    sels = [zij == izi for izi in zi]
    return sels

def selection_slices_by_slice(islices, nslices):
    return [islices == i for i in range(nslices)]


#-----------------------------------
#
#-----------------------------------
