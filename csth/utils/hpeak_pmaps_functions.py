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


DFpmap = collections.namedtuple('DFpmaps', ['s1', 's2', 's2i', 's1pmt', 's2pmt'])

#----------------------------------------
# Utilities to deal with pmaps-dataframes
#-------------------------------------

def get_pmaps_gd(filename):
    hdf = pd.HDFStore(filename)
    dat = [hdf['s1'], hdf['s2'], hdf['s2si'], hdf['s1pmt'], hdf['s2pmt']]
    return DFpmap(*dat)

def get_pmaps(filename):
    return DFpmap(*pmio.load_pmaps_as_df(filename))

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


#---------------------------------------
# main functor that creates hit-peaks data dataframes
#
#---------------------------------------

def functor_hits_from_peak(run_number):
    """ This functor creates a function that using peaks from pmaps information in data-dataframes
    returns corrected hit-peak information in data frames
    """

    correction_filename = f"$IC_DATA/maps/kr_corrections_run{run_number}.h5"
    correction_filename = os.path.expandvars(correction_filename)
    calibrate = corrections.Calibration(correction_filename, 'scale')

    datasipm = db.DataSiPM(run_number)
    sipms_xs, sipms_ys = datasipm.X.values, datasipm.Y.values


    def _hits(s1, xs2, xs2i, s1_peak = 0, s2_peak = 0, vdrift = 1.):
        """ returns corrected information of hits-peak in data-dataframes
        input:
            s1  : data-frame with S1 information
            xs2 : data-frame with S2 information
            xs2i: data-frame with S2i information
            s1_peak: number of the S1 peak (default 0)
            s2_peak: number of the S2 peak (default 0)
            vdrift : drift velocity (default 1.)
        output:
            pksum   : data-frame with peak summary information (total E, etc)
            pkslices: data-frame with peak information per slices (E per slices, etc)
            pkhits  : data-frame with peak information per hit (E per hit, etc)
        """

        # select s2
        s2  = xs2 [xs2 .peak == s2_peak]
        s2i = xs2i[xs2i.peak == s2_peak]

        # t0 and s0
        s1e = np.sum(s1.ene[s1.peak == s1.peak])
        t0  = peak_time(s1 [s1.peak == s1_peak])

        # compute quantities per slice
        zs  = list(1.e-3*vdrift*s2.time-t0)
        nzs = len(zs)
        ks  = list(range(nzs))
        e0s = list(s2.ene)

        nsipms = len(np.unique(s2i.nsipm))

        # slices
        slices = selection_slices(nzs, nsipms)

        # number of hits per slice
        ns = [np.sum(s2i.ene[sel]>0) for sel in slices]

        # indes of slices
        K = np.array(nsipms*ks)

        # total charge per slice
        q0s = [np.sum(s2i.ene[sel]) for sel in slices]

        # compute quantities per hit
        X = np.array(sipms_xs[s2i.nsipm])
        Y = np.array(sipms_ys[s2i.nsipm])
        Z = np.array(nsipms*zs)

        # charge and 'energy' per hit
        Q0i  = np.array(s2i.ene.values)
        E0i  = np.array(nsipms*e0s)

        # corrected 'energy' and charge
        Ei, Qi = calibrate(X, Y, Z, None, E0i, Q0i)

        # total charge per slice
        qs = [np.sum(Qi[sel]) for sel in slices]
        qq = np.array(nsipms*qs)

        # event info
        evt, peak = *set(s2.event), *set(s2.peak)

        qq[qq <= 1.] = 1.
        Fi = Qi/qq
        # share the original and corrected energy per hit
        Ei  = Ei *Fi
        E0i = E0i*Fi

        # total energy per slice
        es = np.array([np.sum(Ei[sel]) for sel in slices])
        es[es <= 1.] = 1.

        # average position per slice
        xs = np.array([np.sum(Ei[sel]*X[sel]) for sel in slices])/es
        ys = np.array([np.sum(Ei[sel]*Y[sel]) for sel in slices])/es

        # peak summary
        esum  = np.sum(es)
        if (esum <=1.): esum = 1.
        xsum  = np.sum([xi*ei for xi, ei in zip(xs, es)])/esum
        ysum  = np.sum([yi*ei for yi, ei in zip(ys, es)])/esum
        zsum  = np.sum([zi*ei for zi, ei in zip(zs, es)])/esum
        if (nzs <= 1.): nzs = 1.
        nsum  = np.sum(ns)/(1.*nzs)
        e0sum = np.sum(e0s)
        q0sum = np.sum(q0s)
        qsum  = np.sum(qs)


        pksummary = pd.DataFrame({'event': [evt], 'peak': [peak], 'S1e': [s1e], 't0': [t0],
                                  'nslices': [nzs], 'nsipms' : [nsipms], 'N': [nsum],
                                  'X': [xsum], 'Y': [ysum], 'Z': [zsum],
                                  'Q0': [q0sum], 'Q': [qsum], 'E0': [e0sum], 'E': [esum]})

        # data-frame with slices info
        pkslices = pd.DataFrame({'event': s2.event.values, 'peak': s2.peak.values,
                                 'N':ns, 'X':xs, 'Y': ys, 'Z':zs,
                                 'Q0': q0s, 'E0':e0s, 'Q':qs, 'E': es})

        # data-frame with the hits information
        pkhits = pd.DataFrame({'event': s2i.event.values, 'peak':s2i.peak.values, 'nsipm':s2i.nsipm.values,
                             'K':K, 'X':X, 'Y':Y, 'Z':Z,
                             'Q0':Q0i, 'Q':Qi, 'E0':E0i, 'E': Ei, 'F':Fi})

        return True, pksummary, pkslices, pkhits

    return _hits

#------------------------------------------------------
#
#-------------------------------------------------------

ETable = collections.namedtuple('ETable', ['event', 'peak', 'nslices', 'nsipms', 'sid', 'nhits', 'hid',
                                           't0', 's1e',
                                           'x0', 'y0', 'z0', 'q0', 'e0',
                                           'x' , 'y' , 'z' , 'q' , 'e' ])

def _table(size, nint, ntot):
    items = [np.zeros(size, dtype = int) for i in range(nint)]
    items += [np.zeros(size) for i in range(nint, ntot)]
    return items

def select_event(s1, s2, s2i):

    evts = set(s1.event)
    #print(' number of events ', len(evts))
    evts = [evt for evt in evts if len(set(s1.peak[s1.event == evt])) == 1]
    #evts = evts[:5]
    #print(' number of selected events ', len(evts))
    npks = [len(set(s2.peak[s2.event == evt])) for evt in evts]

    ievts, ipks = [], []
    for i, evt in enumerate(evts):
        for ipk in range(npks[i]):
            ssel     = np.logical_and(s2.event == evt, s2.peak == ipk)
            nslices  = int(np.sum(ssel))
            hsel  = np.logical_and(s2i.event == evt, s2i.peak == ipk)
            nhits = int(np.sum(hsel))
            if (nslices > 0 and nhits >= nslices):
                ievts.append(evt); ipks.append(ipk)
    return ievts, ipks

def create_event_table(s1, s2, s2i):

    evts, pks = select_event(s1, s2, s2i)

    size = int(len(pks))
    #print(evts)
    #print(npks)
    #print(' number of selected peaks ', size)

    etab = ETable(*_table(size, 7, 19))

    eindex, sindex, hindex = 0, 0, 0
    for evt, ipk in zip(evts, pks):
        etab.event[eindex]   = int(evt)
        etab.peak [eindex]   = int(ipk)
        tsel                 = np.logical_and(s1.event == evt, s1.peak == ipk)
        s1e                  = np.sum(s1.ene[tsel])
        etab.s1e[eindex]     = s1e
        if (s1e <= 1.): s1e  = 1.
        etab.t0[eindex]      = 1e-3*np.sum(s1.ene[tsel]*s1.time[tsel])/s1e
        ssel                 = np.logical_and(s2.event == evt, s2.peak == ipk)
        nslices              = int(np.sum(ssel))
        etab.nslices[eindex] = nslices
        etab.sid[eindex]     = sindex
        etab.e0[eindex]      = np.sum(s2.ene[ssel])
        hsel                 = np.logical_and(s2i.event == evt, s2i.peak == ipk)
        nhits                = int(np.sum(hsel))
        etab.nhits[eindex]   = nhits
        etab.hid[eindex]     = hindex
        etab.nsipms[eindex]  = int(nhits/nslices)
        etab.q0[eindex]      = np.sum(s2i.ene[hsel])
        eindex              += 1
        sindex              += nslices
        hindex              += nhits

    return etab


STable = collections.namedtuple('HTable', ['event', 'peak', 'nhits',
                                           'x0', 'y0', 'z0', 'q0', 'e0',
                                           'x' , 'y' , 'z' , 'q' , 'e' ])

def create_event_slices_table(etab, s2, s2i, vdrift=1.):

    nevts =  len(etab.event)
    size  = np.sum(etab.nslices)
    #print(" htab total size ", size)
    stab  = STable(*_table(size, 3, 13))

    for i in range(nevts):
        evt, pk                              = etab.event[i], etab.peak[i]
        t0                                   = etab.t0[i]
        sindex, nslices                      = etab.sid[i]  , etab.nslices[i]
        hindex, nhits                        = etab.hid[i]  , etab.nhits[i]
        tsel                                 = np.logical_and(s2.event == evt, s2.peak == pk)
        stab.event[sindex: sindex + nslices] = evt
        stab.peak [sindex: sindex + nslices] = pk
        stab.z    [sindex: sindex + nslices] = vdrift*(1.e-3*s2.time[tsel]-t0)
        stab.e0   [sindex: sindex + nslices] = s2.ene[tsel]
        tsel                                 = np.logical_and(s2i.event == evt, s2i.peak == pk)
        q0ij                                 = s2i.ene[tsel]
        nsipms                               = int(nhits/nslices)
        slices                               = selection_slices(nslices, nsipms)
        q0i                                  = np.array([np.sum(q0ij[sel]) for sel in slices])
        stab.q0   [sindex: sindex + nslices] = q0i
        ns                                   = np.array([np.sum(q0ij[sel] >0) for sel in slices], dtype=int)
        stab.nhits[sindex: sindex + nslices] = ns

    return stab


HTable = collections.namedtuple('HTable', ['event', 'peak', 'nsipm', 'islice',
                                           'x', 'y', 'z', 'q0', 'q', 'e0', 'e'])

def create_event_hit_table(etab, stab, s2i, xpos, ypos):

    nevts =  len(etab.event)
    #print(' event items ', nevts)
    size  = np.sum(etab.nhits)
    #print(" htab total size ", size)
    htab  = HTable(*_table(size, 4, 11))

    for i in range(nevts):
        evt   , pk      = etab.event[i], etab.peak[i]
        sindex, nslices = etab.sid[i]  , etab.nslices[i]
        hindex, nhits   = etab.hid[i]  , etab.nhits[i]
        nsipms = int(nhits/nslices)
        #print('evt ', evt)
        #print('nslices, nsipms ' , nslices, nsipms)
        #print('hindex,  nhits ', hindex, nhits)

        tsel                              = np.logical_and(s2i.event == evt, s2i.peak == pk)
        htab.event[hindex: hindex+nhits]  = evt
        htab.peak [hindex: hindex+nhits]  = pk
        htab.nsipm[hindex: hindex+nhits]  = s2i.nsipm[tsel]
        htab.x    [hindex: hindex+nhits]  = xpos[s2i.nsipm[tsel]]
        htab.y    [hindex: hindex+nhits]  = ypos[s2i.nsipm[tsel]]
#        htab.z    [hindex: hindex+nhits] = vdrift*(1.e-3*s2i.time[tsel]-t0)
        htab.q0   [hindex: hindex+nhits]  = s2i.ene[tsel]
        htab.e0   [hindex: hindex+nhits]  = 1.
        ssel                              = np.logical_and(stab.event == evt, stab.peak == pk)
        zi                                = stab.z[ssel]
        #print(zi)
        ij                                = np.zeros(nhits)
        zij                               = np.zeros(nhits)
        nsipms                            = int(nhits/nslices)
        slices                            = selection_slices(nslices, nsipms)
        for k, kslice in enumerate(slices):
            try:
                zij[kslice]  = zi[k]
                ij [kslice]  = k
            except:
                print('Error slices, ', zi, k, nslices, nsipms, nhits)
        htab.z     [hindex: hindex+nhits]  = zij
        htab.islice[hindex: hindex+nhits]  = ij
    return htab


def calibrate_tables(etab, stab, htab, calibrate):

    ec, qc = calibrate(htab.x, htab.y, htab.z, None, htab.e0, htab.q0)
    #print(len(ec), len(qc), len(htab.event))
    htab.e[:] = ec[:]
    htab.q[:] = qc[:]
    #htab.q[:] = htab.q0[:]

    nevts =  len(etab.event)
    #print(' event items ', nevts)

    for eindex in range(nevts):
        evt   , pk      = etab.event[eindex], etab.peak[eindex]
        sindex, nslices = etab.sid  [eindex], etab.nslices[eindex]
        hindex, nhits   = etab.hid  [eindex], etab.nhits[eindex]
        nsipms = int(nhits/nslices)
        #print('evt ', evt)
        #print('nslices, nsipms ' , nslices, nsipms)
        #print('hindex,  nhits ', hindex, nhits)

        # sum the charge per slices
        hsel                                 = np.logical_and(htab.event == evt, htab.peak == pk)
        ssel                                 = np.logical_and(stab.event == evt, stab.peak == pk)

        qij                                  = htab.q[hsel]
        slices                               = selection_slices(nslices, nsipms)
        qi                                   = np.array([np.sum(qij[sel]) for sel in slices])
        #print('qi ', qi)
        qi[qi <= 1.] = 1.
        stab.q    [sindex: sindex + nslices] = qi
        etab.q    [eindex]                   = np.sum(qi)

        # corrected energy per hit
        e0i  = stab.e0[ssel]
        q0i  = stab.q0[ssel]
        q0i[q0i <= 1.] = 1.
        eij  = htab.e [hsel]
        e0ij = htab.e0[hsel]
        q0ij = htab.q0[hsel]
        for k, kslice in enumerate(slices):
            eij [kslice]    = eij[kslice] * qij [kslice]*e0i[k]/qi [k]
            e0ij[kslice]    =               q0ij[kslice]*e0i[k]/q0i[k]
        htab.e0 [hindex: hindex + nhits]     = e0ij
        htab.e  [hindex: hindex + nhits]     = eij


        # sum energy per slice
        ei = np.array([np.sum(eij[sel]) for sel in slices])
        #print('ei ', ei)
        ei[ei <= 1.] = 1.
        stab.e  [sindex: sindex + nslices ] = ei
        ee = np.sum(ei)
        if (ee <= 1.): ee = 1.
        etab.e  [eindex]                    = ee

        # compute the average position per slice
        xij                                 = htab.x[hsel]
        yij                                 = htab.y[hsel]

        xi     = np.array([np.sum(xij[sel]*eij[sel])/ei[k] for k,sel in enumerate(slices)])
        yi     = np.array([np.sum(yij[sel]*eij[sel])/ei[k] for k,sel in enumerate(slices)])
        stab.x  [sindex: sindex + nslices] = xi
        stab.y  [sindex: sindex + nslices] = yi

        x0i     = np.array([np.sum(xij[sel]*e0ij[sel])/e0i[k] for k,sel in enumerate(slices)])
        y0i     = np.array([np.sum(yij[sel]*e0ij[sel])/e0i[k] for k,sel in enumerate(slices)])
        stab.x0 [sindex: sindex + nslices] = xi
        stab.y0 [sindex: sindex + nslices] = yi

        etab.x  [eindex] = np.sum(xi*ei)/ee
        etab.y  [eindex] = np.sum(yi*ei)/ee

        ee0      = np.sum(e0i)
        if (ee0 <= 0): ee0 = 1.
        etab.x0 [eindex] = np.sum(xi*e0i)/ee0
        etab.y0 [eindex] = np.sum(yi*e0i)/ee0

        zi     = stab.z [ssel]
        etab.z0 [eindex] = np.sum(zi*ei) /ee
        etab.z  [eindex] = np.sum(zi*e0i)/ee0

    return etab, stab, htab


def dfs_from_tables(etab, stab, htab):

    nsipms = etab.nhits/etab.nslices

    pksum = pd.DataFrame({ 'event'   : etab.event,
                           'peak'    : etab.peak,
                           'nslices' : etab.nslices,
                           'nsipms'  : nsipms,
                           'X0'      : etab.x0,
                           'Y0'      : etab.y0,
                           'Z0'      : etab.z0,
                           'Q0'      : etab.q0,
                           'E0'      : etab.e0,
                           'X'       : etab.x,
                           'Y'       : etab.y,
                           'Z'       : etab.z,
                           'Q'       : etab.q,
                           'E'       : etab.e})


    pkslice = pd.DataFrame({'event'  :  stab.event,
                            'peak'   :  stab.peak,
                            'N'      :  stab.nhits,
                            'X0'     :  stab.x0,
                            'Y0'     :  stab.y0,
                            'Z0'     :  stab.z0,
                            'Q0'     :  stab.q0,
                            'E0'     :  stab.e0,
                            'X'      :  stab.x,
                            'Y'      :  stab.y,
                            'Z'      :  stab.z,
                            'Q'      :  stab.q,
                            'E'      :  stab.e})

    return pksum, pkslice, None


def create_event_dfs(pmaps, xpos, ypos, calibrate):

    etab = create_event_table(pmaps.s1, pmaps.s2, pmaps.s2i)

    stab = create_event_slices_table(etab, pmaps.s2, pmaps.s2i)

    htab = create_event_hit_table(etab, stab, pmaps.s2i, xpos, ypos)

    etab, stab, htab = calibrate_tables(etab, stab, htab, calibrate)

    pksum, pkslice, _ = dfs_from_tables(etab, stab, htab)

    return pksum, pkslice, _
