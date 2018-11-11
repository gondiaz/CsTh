import os
import numpy             as np
import collections       as collections
import pandas            as pd

#import invisible_cities.database.load_db   as db
#import invisible_cities.io.pmaps_io        as pmio

import krcal.dev.corrections               as corrections
import csth .utils.hpeak_tables            as hptab
import csth .utils.pmaps_functions         as pmapsf

Q0MIN  = 6.
VDRIFT = 1.
CALQ   = True

def events_summary(pmaps, runinfo, loc, xpos, ypos, calibrate,
                   q0min = Q0MIN, vdrift = VDRIFT, calq = CALQ, full = False):

    spmaps    = pmapsf.filter_1s1(pmaps)

    nepks     = pmapsf.neventpeaks(spmaps)
    etab      = hptab.event_table(nepks)

    nslices = len(spmaps.s2)                  if full else 0
    nhits   =  np.sum(spmaps.s2i.ene > q0min) if full else 0
    stab = hptab.slice_table(nslices)         if full else None
    htab = hptab.hit_table  (nhits)           if full else None

    eindex, sindex, hindex = 0, 0, 0
    for epk, pmap in pmapsf.eventpeak_iterator(spmaps):
        evt, ipk      = epk

        cc            = event_summary(pmap, calibrate, xpos, ypos, q0min, vdrift, calq, full)
        timestamp     = event_timestamp(runinfo, evt)

        esum = cc[0] if full else cc
        ssum = cc[1] if full else None
        hsum = cc[2] if full else None

        esum.location = loc
        esum.time     = timestamp

        eindex = hptab.set_table(esum, eindex, etab)
        if full:
            sindex = hptab.set_table(ssum, sindex, stab)
            hindex = hptab.set_table(hsum, hindex, htab)

    edf = hptab.df_from_table(etab)
    if (not full): return edf

    sdf = hptab.df_from_table(stab)
    hdf = hptab.df_from_table(htab)
    return edf, sdf, hdf


def event_summary(pmap, calibrate, xpos, ypos,
                  q0min = Q0MIN, vdrift = VDRIFT, calq = CALQ, full = False):

    s1, s2, s2i               = pmap

    esum                      = hptab.event_table(1)
    evt                       = np.unique(s2.event)[0]
    ipk                       = np.unique(s2.peak)[0]
    esum.event, esum.peak     = evt, ipk

    s1e, t0                   = event_s1_info(s1)
    nslices, z0i, e0i         = event_slices(s2, t0, vdrift)
    if (nslices <= 0): return esum

    nhits, noqhits, q0tot, x0ij, y0ij, z0ij, q0ij = event_hits(s2i, z0i, xpos, ypos, q0min)
    if (nhits <= 0)  : return esum

    x0, y0, z0, q0, e0        = hptab.event_eqpoint(e0i, z0i, x0ij, y0ij, q0ij)
    rmax, rbase               = hptab.radius(x0ij, y0ij, x0, y0)
    zmin, zmax                = hptab.zrange(z0i)

    ceij, cqij                = hptab.calibration_factors(x0ij, y0ij, z0ij, calibrate)
    ei, qi, eij, qij          = hptab.hits_energy(e0i, z0i, z0ij, q0ij, ceij, cqij)
    ec                        = np.sum(ei)
    ei, enoq, noqslices       = hptab.slices_energy(e0i, ei)
    x, y, z, qc, e            = hptab.event_eqpoint(ei , z0i, x0ij, y0ij, qij)

    if (q0 <= 1.): q0 = 1.
    q = qc * q0tot/q0

    esum.s1e, esum.t0         = s1e  , t0
    esum.nslices              = nslices
    esum.nhits, esum.noqhits  = nhits, noqhits
    esum.x0, esum.y0, esum.z0 = x0   , y0, z0
    esum.q0, esum.e0          = q0   , e0
    esum.rmax, esum.rbase     = rmax , rbase
    esum.zmin, esum.zmax      = zmin , zmax
    esum.noqslices            = noqslices
    esum.x , esum.y, esum.z   = x    , y, z
    esum.q , esum.e           = q    , e
    esum.qc, esum.ec          = qc   , ec

    if (not full): return esum

    _, q0i, e0ij, _ = hptab.hits_energy(e0i, z0i, z0ij, q0ij)

    ssum = hptab.slice_table(nslices)
    ssum.event[:], ssum.peak[:] = evt, ipk
    ssum.z0[:]                  = z0i
    ssum.e0[:], ssum.e[:]       = e0i, ei
    ssum.q0[:], ssum.q[:]       = q0i, qi

    hsum = hptab.hit_table(nhits)
    hsum.event[:], hsum.peak[:]         = evt, ipk
    hsum.x0[:], hsum.y0[:], hsum.z0[:]  = x0ij, y0ij, z0ij
    hsum.q0[:], hsum.q[:]               = q0ij, qij
    hsum.e0[:], hsum.e[:]               = e0ij, eij

    return esum, ssum, hsum


def get_event_hits(pmap, calibrate, xpos, ypos, q0min = Q0MIN, vdrift = VDRIFT, calq = CALQ):

    s1, s2, s2i  = pmap

    s1e, t0      = event_s1_info(s1)

    nslices, z0i, e0i  = event_slices(s2, t0, vdrift)
    if (nslices <= 0): return None

    nhits, noqhits, qtot, x0ij, y0ij, z0ij, q0ij = event_hits(s2i, z0i, xpos, ypos, q0min)
    if (nhits <= 0): return None

    noqslices, ei, qi, eij, qij  = hptab.calibrate_hits(e0i, z0i, x0ij, y0ij, z0ij, q0ij, calibrate, calq)

    return x0ij, y0ij, z0ij, eij, qij


def event_s1_info(s1):

    s1e                  = np.sum(s1.ene)
    if (s1e <= 1.): s1e  = 1.
    t0                   = 1e-3*np.sum(s1.ene*s1.time)/s1e
    #print('s1e ', s1e)
    #print('t0  ', t0)
    #print('time', time)
    return s1e, t0


def event_timestamp(runinfo, evt):
    timestamp = runinfo[runinfo.evt_number == evt].timestamp.values[0]
    # print(timestamp)
    return timestamp

def event_slices(s2, t0, vdrift = VDRIFT):

    # z0s and index-slice in slices
    ts  = 1.e-3*s2.time.values
    nslices = len(ts)
    if (nslices <= 0):
        return nslices, None, None
    z0i  = vdrift*(ts-t0)
    e0i  = s2.ene.values

    #print('nslices ', nslices)
    #print('e0i ', len(e0i), np.sum(e0i) , e0i)
    #print('z0i ', len(z0i), np.sum(z0i*e0i)/np.sum(e0i), z0i)
    return nslices, z0i, e0i

def event_hits(s2i, z0i, xpos, ypos, q0min = Q0MIN):

    nslices      = len(z0i)
    if (nslices <= 1):
        return 0, 0, 0, None, None, None, None
    q0ij         = s2i.ene.values
    ntotal_hits  = len(q0ij)
    if (ntotal_hits <= 0):
        return 0, 0, 0, None, None, None, None
    #z0ij         = np.zeros(ntotal_hits)
    nsipms       = int(ntotal_hits/nslices)
    assert int(nsipms*nslices) == ntotal_hits

    qtot  = np.sum(q0ij)
    #
    #selslices    = hptab.selection_slices(nslices, nsipms)
    #for k, kslice in enumerate(selslices):
    #    z0ij[kslice] = z0i[k]
    z0ij = np.tile(z0i, nsipms)

    # get the x, y positions and charge of the siPMs
    qsel    = q0ij > q0min
    noqsel  = (q0ij > 0) & (q0ij <= q0min)
    nhits   = np.sum(qsel)
    noqhits = np.sum(noqsel)
    if (nhits <= 0):
        return nhits, noqhits, qtot, None, None, None, None

    sipm   = s2i.nsipm.values
    q0ij   = q0ij[qsel]
    x0ij   = xpos[sipm[qsel]]
    y0ij   = ypos[sipm[qsel]]
    z0ij   = z0ij[qsel]

    #print('nhits, noqhits ', nhits, noqhits)
    #print('x0ij', len(x0ij), x0ij, '\n y0ij', len(y0ij), y0ij, '\n z0ij', len(z0ij), z0ij)
    #print('q0ij', len(q0ij), q0ij)
    return nhits, noqhits, qtot, x0ij, y0ij, z0ij, q0ij
