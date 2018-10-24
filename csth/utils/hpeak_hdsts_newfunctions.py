import os
import numpy             as np
import collections       as collections
import pandas            as pd

#import invisible_cities.database.load_db   as db
#import invisible_cities.io.pmaps_io        as pmio

import krcal.dev.corrections               as corrections
import csth .utils.hpeak_tables            as hptab

Q0MIN = 6.
NEVARS = len(hptab.edf_names)

def events_summary(hits, loc, calibrate, q0min = Q0MIN):

    elist = event_list(hits)
    evts, npks  = elist

    size = int(np.sum(npks))
    etab = hptab.create_event_table(size)

    eindex = 0
    for evt, npk  in zip(evts, npks):
        for ipk in range(npk):
            hsel = (hits.event == evt) & (hits.npeak == ipk)
            esum = event_summary(hits[hsel], evt, ipk, loc, calibrate, q0min)

            hptab._etable_set(etab, esum, eindex)
            eindex += 1
    edf = hptab.df_from_etable(etab)
    return edf


def event_list(hits):
    evts = np.unique(hits.event)
    #print(' number of events ', len(evts))
    npks = [len(np.unique(hits.npeak[hits.event == evt])) for evt in evts]

    #print('evts ', evts)
    return hptab.EventList(evts, npks)

def event_summary(hits, evt, ipk, loc, calibrate, q0min = Q0MIN):

    nslices, nhits          = 0, 0
    noqslices, noqhits      = 0, 0
    time, s1e, t0           = 0., 0., 0.
    rmax, zmin, zmax        = 0., 0., 0.
    x0, y0, z0, q0, e0      = 0., 0., 0., 0., 0.
    x, y, z, q, e           = 0., 0., 0., 0., 0.

    def _result():
        esum = hptab.ETuple(evt, ipk, loc, nslices, nhits, noqslices, noqhits,
                            time, s1e, t0, rmax, zmin, zmax,
                            x0, y0, z0, q0, e0,
                            x, y, z, q, e)
        #print(esum)
        return esum

    nslices, z0i, e0i      = event_slices(hits)
    if (nslices <= 0):
        return _result()

    nhits, noqhits, x0ij, y0ij, z0ij, q0ij = event_hits(hits, z0i, q0min)
    if (nhits <= 0):
        return _result()

    s1e, t0, time          = event_s1_info(hits)

    x0, y0, z0, q0, e0     = hptab.event_eqpoint(e0i, z0i, x0ij, y0ij, q0ij)

    rmax                   = hptab.max_radius_hit(x0ij, y0ij)

    zmin, zmax             = hptab.zrange(z0i)

    noqslices, ei, qij, eij           = hptab.calibrate_hits(e0i, z0i, x0ij, y0ij, z0ij, q0ij, calibrate)

    x , y , z , q , e      = hptab.event_eqpoint(ei , z0i, x0ij, y0ij, eij)

    esum = _result()
    return esum

def event_s1_info(hits):

    s1e  = 0.
    t0   = 0.
    time = np.unique(hits.time)[0]

    #print('s1e , t0, time ', s1e, t0, time)
    return 0., 0., time

def event_slices(hits):

    z0ij    = hits.Z.values
    z0i     = np.unique(z0ij)
    nslices = len(z0i)
    if (nslices <= 0):
        return nslices, None, None

    e0ij        = hits.E.values
    selslices = hptab.selection_slices_by_z(z0ij, z0i)
    e0i  = np.array([np.sum(e0ij[sel]) for sel in selslices])

    #print('nslices ', nslices)
    #print('e0i ', len(e0i) , np.sum(e0i)  , e0i)
    #print('z0i ', len(z0i),  np.sum(z0i*e0i)/np.sum(e0i), z0i)
    return nslices, z0i, e0i

def event_hits(hits, z0i, q0min = 6.):

    ntot  = len(hits.Q)
    qsel  = hits.Q > q0min
    nhits = int(np.sum(qsel))
    noqhits = ntot - nhits

    if (nhits <= 0):
        return nhits, noqhits, None, None, None, None

    q0ij   = hits.Q[qsel].values
    x0ij   = hits.X[qsel].values
    y0ij   = hits.Y[qsel].values
    z0ij   = hits.Z[qsel].values

    #print('nhits ', nhits, 'noqhits', noqhits)
    #print('x0ij', len(x0ij), x0ij, '\n y0ij', len(y0ij), y0ij, '\n z0ij', len(z0ij), z0ij)
    #print('q0ij', len(q0ij), q0ij)
    return nhits, noqhits, x0ij, y0ij, z0ij, q0ij
