import os
import numpy             as np
import collections       as collections
import pandas            as pd

#import invisible_cities.database.load_db   as db
#import invisible_cities.io.pmaps_io        as pmio

import krcal.dev.corrections               as corrections
import csth .utils.hpeak_tables            as hptab


#-----------------------------------
#    Events table
#-----------------------------------


def hdst_event_list(hits):
    evts = np.unique(hits.event)
    #print(' number of events ', len(evts))
    npks = [len(np.unique(hits.npeak[hits.event == evt])) for evt in evts]

    ievts, ipks = [], []
    for i, evt in enumerate(evts):
        for ipk in range(npks[i]):
            hsel  = np.logical_and(hits.event == evt, hits.npeak == ipk)
            nhits = int(np.sum(hsel))
            if (nhits > 0):
                ievts.append(evt); ipks.append(ipk)
    return hptab.EventList(ievts, ipks)


def hdst_event_table(elist, hits):

    evts, pks = elist.event, elist.peak

    size = int(len(pks))

    etab = hptab.create_event_table(size)

    eindex, sindex, hindex = 0, 0, 0
    for evt, ipk in zip(evts, pks):
        etab.event  [eindex] = int(evt)
        etab.peak   [eindex] = int(ipk)
        etab.s1e    [eindex] = 1.
        hsel                 = np.logical_and(hits.event == evt, hits.npeak == ipk)
        etab.time   [eindex] = np.unique(hits.time [hsel])[0]
        etab.x0     [eindex] = np.unique(hits.Xpeak[hsel])[0]
        etab.y0     [eindex] = np.unique(hits.Ypeak[hsel])[0]
        etab.nslices[eindex] = len(np.unique(hits.Z[hsel]))
        etab.sid    [eindex] = sindex
        nhits                = int(np.sum(hsel))
        etab.nhits  [eindex] = nhits
        etab.hid    [eindex] = hindex
        zij                  = hits.Z[hsel]
        nslices              = len(np.unique(zij))
        e0ij                 = hits.E[hsel]
        e0                   = np.sum(e0ij)
        z                    = np.sum(e0ij*zij)/e0
        etab.e0     [eindex] = e0
        etab.z0     [eindex] = z
        q0ij                 = hits.Q[hsel]
        etab.noqhits[eindex] = np.sum(q0ij <= 0.)
        etab.q0     [eindex] = np.sum(q0ij[q0ij >0])
        eindex              += 1
        sindex              += nslices
        hindex              += nhits

    return etab

#---------------------------------
#   Slices table
#---------------------------------

def hdst_slice_table(etab, hits):

    nevts =  len(etab.event)
    size  = np.sum(etab.nslices)
    #print(" htab total size ", size)
    stab  = hptab.create_slice_table(size)

    for i in range(nevts):
        evt, pk                              = etab.event[i], etab.peak[i]
        sindex, nslices                      = etab.sid[i]  , etab.nslices[i]
        hindex, nhits                        = etab.hid[i]  , etab.nhits[i]
        hsel                                 = np.logical_and(hits.event == evt, hits.npeak == pk)
        stab.event[sindex: sindex + nslices] = evt
        stab.peak [sindex: sindex + nslices] = pk
        zij                                  = hits.Z[hsel]
        zi                                   = np.unique(zij)
        zi.sort()
        selslices                            = hptab.selection_slices_by_z(zij)
        stab.slice[sindex: sindex + nslices] = range(nslices)
        stab.z0   [sindex: sindex + nslices] = zi
        e0ij                                 = hits.E[hsel]
        stab.e0   [sindex: sindex + nslices] = np.array([np.sum(e0ij[sel])   for sel in selslices])
        q0ij                                 = hits.Q[hsel]
        selnoq = q0ij <= 0.
        q0ij[selnoq]                         = 0.
        ns                                   = np.array([np.sum(q0ij[sel]>0) for sel in selslices], dtype=int)
        stab.nhits[sindex: sindex + nslices] = ns
        q0i                                  = np.array([np.sum(q0ij[sel])   for sel in selslices])
        stab.q0   [sindex: sindex + nslices] = q0i
        q0i[q0i <= 1.]                       = 1.
        q0ij[selnoq]                         = 1.
        x0ij                                 = hits.X[hsel]
        x0ij[selnoq]                         = hits.Xpeak[hsel].values[selnoq]
        stab.x0   [sindex: sindex + nslices] = np.array([np.sum(q0ij[sel]*x0ij[sel]) for sel in selslices])/q0i
        y0ij                                 = hits.Y[hsel]
        y0ij[selnoq]                         = hits.Ypeak[hsel].values[selnoq]
        stab.y0   [sindex: sindex + nslices] = np.array([np.sum(q0ij[sel]*y0ij[sel]) for sel in selslices])/q0i

    return stab

#-----------------------------
#    Hits table
#-----------------------------


def hdst_hit_table(etab, hits):

    nevts =  len(etab.event)

    #print(' event items ', nevts)
    size  = np.sum(etab.nhits)

    #print(" htab total size ", size)
    htab  = hptab.create_hit_table(size)

    for i in range(nevts):
        evt   , pk      = etab.event[i], etab.peak[i]
        sindex, nslices = etab.sid[i]  , etab.nslices[i]
        hindex, nhits   = etab.hid[i]  , etab.nhits[i]

        hsel                              = np.logical_and(hits.event == evt, hits.npeak == pk)
        htab.event[hindex: hindex+nhits]  = evt
        htab.peak [hindex: hindex+nhits]  = pk
        htab.nsipm[hindex: hindex+nhits]  = hits.nsipm[hsel]
        q0ij                              = hits.Q[hsel]
        selnoq                            = q0ij <= 0.
        q0ij[selnoq]                      = 0.
        xp                                = np.unique(hits.Xpeak[hsel])[0]
        yp                                = np.unique(hits.Ypeak[hsel])[0]
        x0ij                              = hits.X[hsel]
        y0ij                              = hits.Y[hsel]
        x0ij[selnoq]                      = xp
        y0ij[selnoq]                      = yp
        htab.x0   [hindex: hindex+nhits]  = x0ij
        htab.y0   [hindex: hindex+nhits]  = y0ij
        htab.z0   [hindex: hindex+nhits]  = hits.Z[hsel]
        htab.q0   [hindex: hindex+nhits]  = q0ij
        htab.e0   [hindex: hindex+nhits]  = hits.E[hsel]
        zij                               = hits.Z[hsel]
        ij                                = np.zeros(nhits)
        selslices                         = hptab.selection_slices_by_z(zij)
        for k, kslice in enumerate(selslices):
            ij [kslice]  = k
        htab.slice[hindex: hindex+nhits]  = ij
    return htab


#-----------------------------
#   calibrate hits
#-----------------------------


def hdst_calibrate_hits(htab, calibrate):

    x  = htab.x0
    y  = htab.y0
    z  = htab.z0

    size     = len(htab.q0)
    e1       = np.ones(size)
    q0       = htab.q0

    ec, qc = calibrate(x, y, z, None, e1, q0)

    htab.e[:] = ec[:]
    htab.q[:] = qc[:]

    return htab


#----------------------------------------
#   Update the tables
#----------------------------------------

def hdst_update_tables(etab, stab, htab):

    evts = etab.event

    for eindex in range(len(evts)):
        evt   , pk      = etab.event[eindex], etab.peak[eindex]
        sindex, nslices = etab.sid  [eindex], etab.nslices[eindex]
        hindex, nhits   = etab.hid  [eindex], etab.nhits[eindex]

        # sum the charge per slices
        hsel                                 = np.logical_and(htab.event == evt, htab.peak == pk)
        ssel                                 = np.logical_and(stab.event == evt, stab.peak == pk)

        qij                                  = htab.q[hsel]
        islices                              = htab.slice[hsel]
        selslices                            = hptab.selection_slices_by_slice(islices, nslices)
        qi                                   = np.array([np.sum(qij[sel]) for sel in selslices])
        stab.q    [sindex: sindex + nslices] = qi
        etab.q    [eindex]                   = np.sum(qi)

        # corrected energy per hit
        qij[qij <= 1.]                       = 1.
        qi [qi  <= 1.]                       = 1.
        eij  = htab.e [hsel]
        e0i  = stab.e0[ssel]
        for k, kslice in enumerate(selslices):
            eij [kslice]                    *= e0i[k] * qij [kslice]/qi[k]
        htab.e  [hindex: hindex + nhits]     = eij

        # sum energy per slice
        ei = np.array([np.sum(eij[sel]) for sel in selslices])
        #print('ei ', ei)
        stab.e  [sindex: sindex + nslices ] = ei
        ee = np.sum(ei)
        etab.e  [eindex]                    = ee

        # compute the average position per slice
        ei[ei <= 1.] = 1.
        if (ee <= 1.):
            ee = 1.

        # compute the average position per slice
        x0ij   = htab.x0[hsel]
        y0ij   = htab.y0[hsel]

        xi     = np.array([np.sum(x0ij[sel]*eij[sel])/ei[k] for k, sel in enumerate(selslices)])
        yi     = np.array([np.sum(y0ij[sel]*eij[sel])/ei[k] for k, sel in enumerate(selslices)])
        stab.x  [sindex: sindex + nslices] = xi
        stab.y  [sindex: sindex + nslices] = yi

        # compute average position per event
        etab.x  [eindex] = np.sum(xi*ei)/ee
        etab.y  [eindex] = np.sum(yi*ei)/ee
        z0i              = stab.z0 [ssel]
        etab.z [eindex]  = np.sum(z0i*ei)/ee

    return etab, stab, htab

#------------------------
#    Main driver
#------------------------


def hdst_convert_to_dfs(hits, calibrate):

    elist = hdst_event_list(hits)

    etab  = hdst_event_table(elist, hits)

    stab  = hdst_slice_table(etab, hits)

    htab  = hdst_hit_table(etab, hits)

    htab  = hdst_calibrate_hits(htab, calibrate)

    etab, stab, htab = hdst_update_tables(etab, stab, htab)

    edf = hptab.df_from_etable(etab)

    sdf = hptab.df_from_stable(stab)

    hdf = hptab.df_from_htable(htab)

    return edf, sdf, hdf
