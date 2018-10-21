import os
import numpy             as np
import collections       as collections
import pandas            as pd

#import invisible_cities.database.load_db   as db
#import invisible_cities.io.pmaps_io        as pmio

import krcal.dev.corrections               as corrections
import csth .utils.hpeak_tables            as hptab


Q0MIN  = 6.
VDRIFT = 1.

def events_summary(pmaps, xpos, ypos, calibrate, q0min = Q0MIN, vdrift = VDRIFT):

    elist = event_list(pmaps)
    evts, npks  = elist

    size = int(np.sum(npks))
    etab = hptab.create_event_table(size)

    eindex = 0
    for evt, npk  in zip(evts, npks):
        for ipk in range(npk):
            esum = event_summary(pmaps, evt, ipk, calibrate, xpos, ypos, q0min, vdrift)

            hptab._etable_set(etab, esum, eindex)
            eindex += 1

    edf = hptab.df_from_etable(etab)
    return edf

def event_list(pmaps):

    s1, s2, s2i = pmaps.s1, pmaps.s2, pmaps.s2i

    evts = np.unique(s1.event)
    evts = [evt for evt in evts if len(np.unique(s1.peak[s1.event == evt])) == 1]
    npks = [len(np.unique(s2.peak[s2.event == evt])) for evt in evts]

    #print('event ', evts)
    #print('peaks ', npks)
    return hptab.EventList(evts, npks)


def event_summary(pmaps, evt, ipk, calibrate, xpos, ypos, q0min = Q0MIN, vdrift = VDRIFT):

    s1, s2, s2i = pmaps.s1, pmaps.s2, pmaps.s2i

    tsel                 = (s1.event == evt)  & (s1.peak == 0)
    ssel                 = (s2.event == evt)  & (s2.peak == ipk)
    hsel                 = (s2i.event == evt) & (s2i.peak == ipk)

    time, s1e, t0          = event_s1_info(s1[tsel])

    nslices, z0i, e0i       = event_slices(s2[ssel], t0, vdrift)
    if (nslices <= 0):
        esum = hptab.ETuple(evt, ipk, nslices, 0, 0,
                            0, 0, 0, 0,
                            0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0)
        return esum

    nhits, noqhits, x0ij, y0ij, z0ij, q0ij = event_hits(s2i[hsel], z0i, xpos, ypos, q0min)
    if (nhits <= 0):
        esum = hptab.ETuple(evt, ipk, nslices, nhits, noqhits,
                                0, 0, 0, 0,
                                0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0)
        return esum

    x0, y0, z0, q0, e0     = hptab.event_eqpoint(e0i, z0i, x0ij, y0ij, q0ij)

    rmax                   = hptab.max_radius_hit(x0ij, y0ij)

    ei, qij, eij           = hptab.calibrate_hits(e0i, z0i, x0ij, y0ij, z0ij, q0ij, calibrate)

    x , y ,  z, q , e      = hptab.event_eqpoint(ei , z0i, x0ij, y0ij, eij)

    esum = hptab.ETuple(evt, ipk, nslices, nhits, noqhits,
                        time, s1e, t0, rmax,
                        x0, y0, z0, q0, e0,
                        x, y, z, q, e)
    #print(esum)
    return esum


def event_s1_info(s1):

    s1e                  = np.sum(s1.ene)
    if (s1e <= 1.): s1e  = 1.
    t0                   = 1e-3*np.sum(s1.ene*s1.time)/s1e

    #print('s1e ', s1e)
    #print('t0  ', t0)
    return 0., s1e, t0

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
        return 0, 0, None, None, None, None
    q0ij         = s2i.ene.values
    ntotal_hits  = len(q0ij)
    if (ntotal_hits <= 0):
        return 0, 0, None, None, None, None
    z0ij         = np.zeros(ntotal_hits)
    nsipms       = int(ntotal_hits/nslices)
    assert int(nsipms*nslices) == ntotal_hits
    selslices    = hptab.selection_slices(nslices, nsipms)
    for k, kslice in enumerate(selslices):
        z0ij[kslice] = z0i[k]

    # get the x, y positions and charge of the siPMs
    qsel   = q0ij > q0min

    nhits = np.sum(qsel)
    noqhits = ntotal_hits - nhits
    if (nhits <= 0):
        return nhits, noqhits, None, None, None, None

    sipm   = s2i.nsipm.values
    q0ij   = q0ij[qsel]
    x0ij   = xpos[sipm[qsel]]
    y0ij   = ypos[sipm[qsel]]
    z0ij   = z0ij[qsel]

    #print('nhits, noqhits ', nhits, noqhits)
    #print('x0ij', len(x0ij), x0ij, '\n y0ij', len(y0ij), y0ij, '\n z0ij', len(z0ij), z0ij)
    #print('q0ij', len(q0ij), q0ij)
    return nhits, noqhits, x0ij, y0ij, z0ij, q0ij

event_df_fast = events_summary
