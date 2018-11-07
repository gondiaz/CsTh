import os
import numpy             as np
import collections       as collections
import pandas            as pd

import krcal.dev.corrections               as corrections
import csth .utils.hpeak_tables            as hptab

Q0MIN  = 6.
CALQ   = True

def events_summary(hits, loc, calibrate, q0min = Q0MIN, calq = CALQ):

    groups = hits.groupby(['event', 'npeak'])

    nepks  = len(groups)
    etab = hptab.event_table(nepks)

    eindex = 0
    for epk, ehits  in groups:
        esum = event_summary(ehits, calibrate, q0min, calq)
        esum.location = loc
        eindex = hptab.set_table(esum, eindex, etab)
    edf = hptab.df_from_table(etab)
    return edf


def event_summary(hits, calibrate, q0min = Q0MIN, calq = CALQ):

    esum = hptab.event_table(1)

    evt                       = np.unique(hits.event)[0]
    ipk                       = np.unique(hits.npeak)[0]
    esum.event, esum.peak     = evt, ipk

    s1e, t0, timestamp        = event_s1_info(hits)
    esum.s1e, esum.t0         = s1e, t0
    esum.time                 = timestamp

    nslices, z0i, e0i      = event_slices(hits)
    esum.nslices           = nslices
    if (nslices <= 0): return esum

    nhits, noqhits, x0ij, y0ij, z0ij, q0ij = event_hits(hits, z0i, q0min)
    esum.nhits, esum.noqhits  = nhits, noqhits
    if (nhits <= 0):   return esum

    s1e, t0, timestamp        = event_s1_info(hits)
    esum.s1e, esum.t0         = s1e, t0
    esum.time                 = timestamp

    x0, y0, z0, q0, e0      = hptab.event_eqpoint(e0i, z0i, x0ij, y0ij, q0ij)
    esum.x0, esum.y0, esum.z0 = x0, y0, z0
    esum.q0, esum.e0          = q0, e0

    rmax                    = hptab.max_radius_hit(x0ij, y0ij)
    esum.rmax = rmax

    zmin, zmax              = hptab.zrange(z0i)
    esum.zmin, esum.zmax    = zmin, zmax

    noqslices, ei, qij, eij = hptab.calibrate_hits(e0i, z0i, x0ij, y0ij, z0ij, q0ij, calibrate, calq)
    esum.noqslices          = noqslices

    x, y, z, q, e           = hptab.event_eqpoint(ei , z0i, x0ij, y0ij, eij)
    esum.x, esum.y, esum.z    = x, y, z
    esum.q, esum.e            = z, e

    return esum


def get_event_hits(hits, calibrate, q0min = Q0MIN, calq = CALQ):

    nslices, z0i, e0i      = event_slices(hits)
    if (nslices <= 0):
        return None

    nhits, noqhits, x0ij, y0ij, z0ij, q0ij = event_hits(hits, z0i, q0min)
    if (nhits <= 0):
        return None

    noqslices, ei, qij, eij  = hptab.calibrate_hits(e0i, z0i, x0ij, y0ij, z0ij, q0ij, calibrate, calq)

    return x0ij, y0ij, z0ij, eij


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
