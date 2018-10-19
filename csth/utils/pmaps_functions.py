import numpy             as np
import collections       as collections
import pandas            as pd

import invisible_cities.io.pmaps_io        as pmio

DFpmap = collections.namedtuple('DFpmaps', ['s1', 's2', 's2i', 's1pmt', 's2pmt'])

#----------------------------------------
# Utilities to deal with pmaps-dataframes
#-------------------------------------


def get_pmaps(filename, mode = ''):
    if (mode == 'gd'):
        hdf = pd.HDFStore(filename)
        dat = [hdf['s1'], hdf['s2'], hdf['s2si'], hdf['s1pmt'], hdf['s2pmt']]
        return DFpmap(*dat)
    dat = pmio.load_pmaps_as_df(filename)
    return DFpmap(*dat)

#def get_pmaps(filename):
#    return DFpmap(*pmio.load_pmaps_as_df(filename))

def pmaps_event_list(dfs):
    s1events = set(dfs.s1.event)
    s2events = set(dfs.s2.event)
    xevents = s1events.union(s2events)
    return xevents

def pmaps_get_event(dfs, event):
    pm = DFpmap(dfs.s1   [dfs.s1   .event == event],
                dfs.s2   [dfs.s2   .event == event],
                dfs.s2i  [dfs.s2i  .event == event],
                dfs.s1pmt[dfs.s1pmt.event == event],
                dfs.s2pmt[dfs.s2pmt.event == event])
    return pm

def pmap_npeaks(pm):
    ns1 = len(set(pm.s1.peak))
    ns2 = len(set(pm.s2.peak))
    return ns1, ns2

def pmap_times(pm, s0_peak = 0, s2_peak = 0):
    t0 = peak_time(pm.s1[pm.s1.peak == s0_peak])
    t1 = peak_time(pm.s2[pm.s2.peak == s2_peak])
    return t0, t1, t1-t0

def peak_time(peak):
    return 1.e-3*np.sum(peak.time*peak.ene)/np.sum(peak.ene)

def peak_time_width(peak):
    t0, ti = np.min(peak.time), np.max(peak.time)
    return t0, ti, ti-t0

def pmap_hits(pm, s0_peak=0, s2_peak=0, vdrift = 1.):
    t0, t1, dt = pmap_times(pm, s0_peak, s2_peak)
    si = pm.s2i[pm.s2i.peak == s2i_peak]

def pmap_slices(pm):
    nsipms = len(np.unique(pm.s2i.nsipm))
    nzs    = len(pm.s2.time)
    #print('nslices, nsipms', nzs, nsipms)
    return nzs, nsipms
