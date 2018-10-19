#   Functions to deal with pmaps-dataframes
#
#   J.A. Hernando 10/10/18
#

import os
import numpy             as np
import collections       as collections
import pandas            as pd

import invisible_cities.database.load_db   as db
import invisible_cities.io.pmaps_io        as pmio

import krcal.dev.corrections               as corrections

import csth.utils.hpeak_tables             as hptab
import csth.utils.pmaps_functions          as pmf

#------------------------------------------------------
#
#-------------------------------------------------------

Q0MIN =  6. # pes
VDRIFT = 1. # mm/us

def event_list(pmaps, q0min = Q0MIN, verbose = False):

    s1, s2, s2i = pmaps.s1, pmaps.s2, pmaps.s2i

    evts = np.unique(s1.event)
    #print(' number of events ', len(evts))
    evts = [evt for evt in evts if len(np.unique(s1.peak[s1.event == evt])) == 1]
    #evts = evts[:5]
    #print(' number of selected events ', len(evts))
    npks = [len(np.unique(s2.peak[s2.event == evt])) for evt in evts]

    """
    ievts, ipks, islices, ihits = [], [], [], []
    for i, evt in enumerate(evts):
        for ipk in range(npks[i]):
            ssel     = (s2.event == evt) & (s2.peak == ipk)
            nslices  = int(np.sum(ssel))
            hsel  =  (s2i.event == evt) & (s2i.peak == ipk) & (s2i.ene > q0min)
            nhits = int(np.sum(hsel))
            if ((nslices > 0) & (nhits > 0)):
                ievts  .append(evt)
                ipks   .append(ipk)
                islices.append(nslices)
                ihits  .append(nhits)
    """

    #print(' event  selected '    , len(set(evts)))
    #print(' peaks  selected '    , np.sum(npks))
    #print(' size of slice table ', np.sum(islices))
    #print(' size of hit table '  , np.sum(ihits))
    return hptab.EventList(evts, npks)

def event_table(evtlist, pmaps, q0min = Q0MIN):

    s1, s2, s2i = pmaps.s1, pmaps.s2, pmaps.s2i

    evts, npks = evtlist.event, evtlist.peak

    size = int(np.sum(npks))
    #print(evts)
    #print(npks)
    #print(' number of selected peaks ', size)

    etab = hptab.create_event_table(size)

    eindex, sindex, hindex = 0, 0, 0
    for evt, npk in zip(evts, npks):
        for ipk in range(npk):
            etab.event[eindex]   = int(evt)
            etab.peak [eindex]   = int(ipk)
            tsel                 = (s1.event == evt) & (s1.peak == ipk)
            s1e                  = np.sum(s1.ene[tsel])
            etab.s1e[eindex]     = s1e
            if (s1e <= 1.): s1e  = 1.
            etab.t0[eindex]      = 1e-3*np.sum(s1.ene[tsel]*s1.time[tsel])/s1e
            ssel                 = (s2.event == evt) & (s2.peak == ipk)
            nslices              = int(np.sum(ssel))
            etab.nslices[eindex] = nslices
            etab.sid[eindex]     = sindex
            etab.e0[eindex]      = np.sum(s2.ene[ssel])
            hsel                 = (s2i.event == evt) & (s2i.peak == ipk)
            ntotal_hits          = np.sum(hsel)
            q0ij                 = s2i.ene[hsel]
            qsel                 = q0ij > q0min
            etab.q0[eindex]      = np.sum(q0ij[hsel])
            nhits                = int(np.sum(qsel))
            etab.nhits[eindex]   = nhits
            etab.noqhits[eindex] = ntotal_hits - nhits
            etab.hid[eindex]     = hindex
            #print('event, pk ', evt, ipk)
            #print('nslices, nhits', nslices, nhits)
    #        etab.nsipms[eindex]  = int(nhits/nslices)
            eindex              += 1
            sindex              += nslices
            hindex              += nhits

    return etab


def slice_table(etab, pmaps, vdrift = VDRIFT):

    s2, s2i = pmaps.s2, pmaps.s2i

    nevts =  len(etab.event)
    size  = np.sum(etab.nslices)
    #print(" htab total size ", size)
    stab  = hptab.create_slice_table(size)

    for i in range(nevts):
        evt, pk                              = etab.event[i], etab.peak[i]
        t0                                   = etab.t0[i]
        sindex, nslices                      = etab.sid[i]  , etab.nslices[i]
        hindex, nhits                        = etab.hid[i]  , etab.nhits[i]
        tsel                                 = (s2.event == evt) & (s2.peak == pk)
        stab.event[sindex: sindex + nslices] = evt
        stab.peak [sindex: sindex + nslices] = pk
        stab.slice[sindex: sindex + nslices] = np.arange(nslices)
        stab.z0   [sindex: sindex + nslices] = vdrift*(1.e-3*s2.time[tsel]-t0)
        stab.e0   [sindex: sindex + nslices] = s2.ene[tsel]
        #tsel                                 = np.logical_and(s2i.event == evt, s2i.peak == pk)
        #q0ij                                 = s2i.ene[tsel]
        #nsipms                               = int(nhits/nslices)
        #slices                               = hptab.selection_slices(nslices, nsipms)
        #q0i                                  = np.array([np.sum(q0ij[sel]) for sel in slices])
        #stab.q0   [sindex: sindex + nslices] = q0i
        #ns                                   = np.array([np.sum(q0ij[sel] > Q0MIN) for sel in slices], dtype=int)
        #stab.nhits[sindex: sindex + nslices] = ns

    return stab


def hit_table(etab, stab, pmaps, xpos, ypos, q0min = Q0MIN):

    s2i = pmaps.s2i

    nevts =  len(etab.event)
    #print(' event items ', nevts)
    size  = np.sum(etab.nhits)
    #print(" htab total size ", size)
    htab  = hptab.create_hit_table(size)

    for i in range(nevts):
        evt   , pk      = etab.event[i], etab.peak[i]
        sindex, nslices = etab.sid[i]  , etab.nslices[i]
        hindex, nhits   = etab.hid[i]  , etab.nhits[i]
        #nsipms = int(nhits/nslices)
        #print('evt, pk ', evt, pk)
        #print('nslices, nhits ' , nslices, nhits)
        #print('sindex,  hindex ', sindex,  hindex)

        # selections
        ssel       =  (stab.event == evt) & (stab.peak == pk)
        hsel       =  (s2i.event == evt)  & (s2i.peak == pk)
        qsel       =  (s2i.event == evt)  & (s2i.peak == pk) & (s2i.ene > q0min)
        #print('nslice, ntotal hits, nhits', np.sum(ssel), np.sum(hsel), np.sum(qsel))


        # set the slices and the zs to the hits
        zi                                = stab.z0[ssel]
        ntotal_hits                       = int(np.sum(hsel))
        ij                                = np.zeros(ntotal_hits)
        zij                               = np.zeros(ntotal_hits)
        nsipms                            = int(ntotal_hits/nslices)
        selslices                         = hptab.selection_slices(nslices, nsipms)
        for k, kslice in enumerate(selslices):
            zij[kslice]  = zi[k]
            ij [kslice]  = k

        # select hits with charge above a threshold
        q0ij                              = s2i.ene[hsel]
        sipmij                            = s2i.nsipm[hsel]

        qsel  = q0ij > q0min

        if (int(np.sum(qsel)) != nhits):
            raise Exception('Not equal number of hits in event ', evt, nhits, np.sum(qsel))

        # store hits for this event
        htab.event [hindex: hindex+nhits]  = evt
        htab.peak  [hindex: hindex+nhits]  = pk
        htab.slice [hindex: hindex+nhits]  = ij[qsel]
        sipmij_withq = sipmij[qsel]
        htab.nsipm [hindex: hindex+nhits]  = sipmij_withq
        htab.x0    [hindex: hindex+nhits]  = xpos[sipmij_withq]
        htab.y0    [hindex: hindex+nhits]  = ypos[sipmij_withq]
        htab.z0    [hindex: hindex+nhits]  = zij[qsel]
#        htab.z    [hindex: hindex+nhits] = vdrift*(1.e-3*s2i.time[tsel]-t0)
        htab.q0    [hindex: hindex+nhits]  = q0ij[qsel]
        htab.e0    [hindex: hindex+nhits]  = 1.

    return htab


def calibrate_hits(htab, calibrate):

    ec, qc = calibrate(htab.x0, htab.y0, htab.z0, None, htab.e0, htab.q0)
    #print(len(ec), len(qc), len(htab.event))
    htab.e[:] = ec[:]
    htab.q[:] = qc[:]
    #htab.q[:] = htab.q0[:]

    return htab

def update_tables(etab, stab, htab):


    nevts =  len(etab.event)
    #print(' event items ', nevts)

    for eindex in range(nevts):
        evt   , pk      = etab.event[eindex], etab.peak[eindex]
        sindex, nslices = etab.sid  [eindex], etab.nslices[eindex]
        hindex, nhits   = etab.hid  [eindex], etab.nhits[eindex]
        #print('evt ', evt)
        #print('nslices, nsipms ' , nslices, nsipms)
        #print('hindex,  nhits ', hindex, nhits)

        # sum the charge per slices
        hsel                                 = (htab.event == evt) & (htab.peak == pk)
        ssel                                 = (stab.event == evt) & (stab.peak == pk)

        islices                              = htab.slice[hsel]
        selslices                            = hptab.selection_slices_by_slice(islices, nslices)

        qij                                  = htab.q[hsel]
        qi                                   = np.array([np.sum(qij[sel]) for sel in selslices])
        stab.q    [sindex: sindex + nslices] = qi
        etab.q    [eindex]                   = np.sum(qi)
        qi [qi <= 1.] = 1.

        q0ij                                 = htab.q0[hsel]
        q0i                                  = np.array([np.sum(q0ij[sel]) for sel in selslices])
        stab.q0   [sindex: sindex + nslices] = q0i
        etab.q0   [eindex]                   = np.sum(q0i)
        q0i [q0i <= 1.] = 1.


        # corrected energy per hit
        e0i  = stab.e0[ssel]
        e0ij = htab.e0[hsel]
        eij  = htab.e [hsel]
        #fij  = np.ones(nhits)
        #f0ij = np.ones(nhits)
        for k, kslice in enumerate(selslices):
            eij [kslice]    = eij[kslice] * qij [kslice]*e0i[k]/qi [k]
            e0ij[kslice]    =               q0ij[kslice]*e0i[k]/q0i[k]
        htab.e0  [hindex: hindex + nhits]    = e0ij
        htab.e   [hindex: hindex + nhits]    = eij


        # slices
        ei             = np.array([np.sum(eij[sel]) for sel in selslices])
        e0i [e0i <= 1] = 1.
        fi             = ei/e0i
        selnoq         = fi <= 0.
        fmed           = np.mean(fi[~selnoq])
        ei [selnoq]    = fmed * e0i [selnoq]

        #print('ei ', ei)
        stab.e  [sindex: sindex + nslices ] = ei
        ee = np.sum(ei)
        etab.e  [eindex]                    = ee
        ei [ei <= 1.] = 1.

        # compute the average position per slice
        x0ij                                 = htab.x0[hsel]
        y0ij                                 = htab.y0[hsel]

        xi     = np.array([np.sum(x0ij[sel]*eij[sel])/ei[k] for k,sel in enumerate(selslices)])
        yi     = np.array([np.sum(y0ij[sel]*eij[sel])/ei[k] for k,sel in enumerate(selslices)])
        stab.x  [sindex: sindex + nslices] = xi
        stab.y  [sindex: sindex + nslices] = yi

        x0i     = np.array([np.sum(x0ij[sel]*e0ij[sel])/e0i[k] for k,sel in enumerate(selslices)])
        y0i     = np.array([np.sum(y0ij[sel]*e0ij[sel])/e0i[k] for k,sel in enumerate(selslices)])
        stab.x0 [sindex: sindex + nslices] = xi
        stab.y0 [sindex: sindex + nslices] = yi

        stab.nhits[sindex: sindex + nslices] = np.array([int(np.sum(sel)) for sel in selslices])

        # global position per event

        etab.x  [eindex] = np.sum(xi*ei)/ee
        etab.y  [eindex] = np.sum(yi*ei)/ee

        ee0      = np.sum(e0i)
        if (ee0 <= 0): ee0 = 1.
        etab.x0 [eindex] = np.sum(xi*e0i)/ee0
        etab.y0 [eindex] = np.sum(yi*e0i)/ee0

        zi     = stab.z0 [ssel]
        etab.z0 [eindex] = np.sum(zi*ei) /ee
        etab.z  [eindex] = np.sum(zi*e0i)/ee0

    return etab, stab, htab


def hpeaks_dfs(pmaps, xpos, ypos, calibrate, q0min = Q0MIN, vdrift = VDRIFT):

    elist = event_list(pmaps, q0min)

    etab  = event_table(elist, pmaps, q0min)

    stab  = slice_table(etab, pmaps, vdrift)

    htab  = hit_table(etab, stab, pmaps, xpos, ypos, q0min)

    htab  = calibrate_hits(htab, calibrate)

    etab, stab, htab = update_tables(etab, stab, htab)

    edf = hptab.df_from_etable(etab)

    sdf = hptab.df_from_stable(stab)

    hdf = hptab.df_from_htable(htab)

    return edf, sdf, hdf
    #edf, sdf, hdf    = convert_to_hpeaks_dfs(etab, stab, htab)

    return edf, sdf, hdf


#-------------------------------------------------
# Fast version
#-------------------------------------------------


def event_table_fast(pmaps, xpos, ypos, calibrate, q0min = Q0MIN, vdrift = VDRIFT):


    elist = event_list(pmaps, q0min = q0min)

    evts, npks  = elist
    s1, s2, s2i = pmaps.s1, pmaps.s2, pmaps.s2i

    size = int(np.sum(npks))
    etab = hptab.create_event_table(size)

    print('total number of peaks ', size)

    xtime = np.zeros(size)

    eindex = 0
    for evt, npk  in zip(evts, npks):
        for ipk in range(npk):

            #xtinit = time.time()

            etab.event[eindex] = evt
            etab.peak [eindex] = ipk

            # selections
            tsel                 = (s1.event == evt)  & (s1.peak == 0)
            ssel                 = (s2.event == evt)  & (s2.peak == ipk)
            hsel                 = (s2i.event == evt) & (s2i.peak == ipk)
            qsel                 = s2i.ene[hsel] > q0min

            nslices = int(np.sum(ssel))
            nhits   = int(np.sum(qsel))
            noqhits = int(np.sum(hsel))-nhits
            etab.nslices[eindex] = nslices
            etab.nhits  [eindex] = nhits
            etab.noqhits[eindex] = noqhits
            if (nslices <= 0 or nhits <= 0): continue

            # s1 information
            s1e                  = np.sum(s1.ene[tsel])
            etab.s1e[eindex]     = s1e
            #print('s1e ', s1e)

            if (s1e <= 1.): s1e  = 1.
            t0                   = 1e-3*np.sum(s1.ene[tsel]*s1.time[tsel])/s1e
            etab.t0[eindex]      = t0
            #print('t0 ', t0)

            # z0s and index-slice in slices
            z0i                  = vdrift*(1.e-3*s2.time[ssel].values-t0)
            #print('z0i', len(z0i), z0i)

            ntotal_hits          = int(np.sum(hsel))
            ij                   = np.zeros(ntotal_hits)
            z0ij                 = np.zeros(ntotal_hits)
            nsipms               = int(ntotal_hits/nslices)
            selslices            = hptab.selection_slices(nslices, nsipms)
            for k, kslice in enumerate(selslices):
                z0ij[kslice] = z0i[k]
                ij [kslice]  = k

            # get the x, y positions and charge of the siPMs
            sipm                 = s2i.nsipm[hsel].values
            q0ij                 = s2i.ene  [hsel].values

            e0ij                 = np.ones(nhits)
            q0ij                 = q0ij[qsel]
            x0ij                 = xpos[sipm[qsel]]
            y0ij                 = ypos[sipm[qsel]]
            z0ij                 = z0ij[qsel]
            ij                   = ij  [qsel]

            # calibrate the SiPMs
            #print('x0', len(x0ij), x0ij, '\n y0', len(y0ij), y0ij, '\n z0', len(z0ij), z0ij)
            #print('e0', len(e0ij), e0ij, '\n q0', len(q0ij), q0ij)
            #print('slice', len(ij), ij)
            eij, qij             = calibrate(x0ij, y0ij, z0ij, None, e0ij, q0ij)
            #print('q ', len(qij), qij)
            #print('e ', len(eij), eij)

            # compute the q0i (total charge per slice)
            selslices            = hptab.selection_slices_by_slice(ij, nslices)

            qi                   = np.array([np.sum(qij[sel]) for sel in selslices])
            q0                   = np.sum(q0ij)
            qq                   = np.sum(qi)
            etab.q0[eindex]      = q0
            etab.q [eindex]      = qq
            #print('qi ', qi)
            #print('q  ', qq)
            #print('q0 ', q0)
            qi [qi <= 1.]        = 1.

            # corrected energy per hit
            e0i  = s2.ene[ssel].values
            e0   = np.sum(e0i)
            #print('e0i ', e0i)
            #print('e0  ', e0)
            etab.e0 [eindex]     = e0
            for k, kslice in enumerate(selslices):
                eij [kslice]     = eij[kslice] * qij [kslice]*e0i[k]/qi [k]

            # compute the energy per slice and the average energy correction per slice
            # to apply to the slices with no charge
            ei             = np.array([np.sum(eij[sel]) for sel in selslices])
            e0i [e0i <= 1] = 1.
            fi             = ei/e0i
            selnoq         = fi <= 0.
            fmed           = np.mean(fi[~selnoq])
            ei [selnoq]    = fmed * e0i [selnoq]
            #print('ei'  , ei)

            ee = np.sum(ei)
            etab.e[eindex] = ee
            #print('e'   , ee)

            # compute the average position
            if (ee <= 1.): ee = 1.
            z     = np.sum(z0i*ei) /ee
            eeave                  = np.sum(eij)
            if (eeave <=1.): eeave = 1.
            x     = np.sum(x0ij*eij)/eeave
            y     = np.sum(y0ij*eij)/eeave
            #print('x, y, z ', x, y, z)

            etab.x[eindex]    = x
            etab.y[eindex]    = y
            etab.z[eindex]    = z

            if (e0 <= 1.): e0 = 1.
            z0    = np.sum(z0i*e0i)/e0
            q0ave = np.sum(q0ij)
            if (q0ave <= 1.): q0ave = 1.
            x0    = np.sum(x0ij*q0ij)/q0ave
            y0    = np.sum(y0ij*q0ij)/q0ave
            #print('x0, y0, z0 ', x0, y0, z0)

            etab.x0[eindex]    = x0
            etab.y0[eindex]    = y0
            etab.z0[eindex]    = z0

            eindex            += 1
            # xtend = time.time()
            # xtime[eindex-1] = xtend - xtinit
            if (eindex % 250 == 0):
                print('processed ', eindex, 'peaks')

            #print('e0i', np.sum(e0i), e0i)
            #print('ei' , np.sum(ei) , ei)
            #print('e0, e', e0, ee, ee/e0)

            #print('q'  , np.sum(qi) , qi)
            #print('q0, q', q0, qq, qq/q0)

    edf = hptab.df_from_etable(etab)
    return edf
