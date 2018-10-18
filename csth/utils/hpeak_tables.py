import os
import numpy             as np
import collections       as collections
import pandas            as pd

#---------------------------
#    Tables definitions
#---------------------------

EventList = collections.namedtuple('EventList', ['event', 'peak'])

etable_names = ['event', 'peak', 'nslices', 'nhits', 'noqhits',
                'sid', 'hid', 'time', 's1e', 't0', 'rmax',
                'x0', 'y0', 'z0', 'q0', 'e0',
                'x' , 'y' , 'z' , 'q' , 'e' ]

edf_names    = ['event', 'peak', 'nslices', 'nhits', 'noqhits',
                'time', 's1e', 't0', 'rmax',
                'x0', 'y0', 'z0', 'q0', 'e0',
                'x', 'y', 'z', 'q', 'e']

ETable = collections.namedtuple('ETable', etable_names)

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
    return ETable(*_table(size, 7, len(etable_names)))

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

def selection_slices_by_z(zij):
    zi = np.unique(zij)
    sels = [zij == izi for izi in zi]
    return sels

def selection_slices_by_slice(islices, nslices):
    return [islices == i for i in range(nslices)]
