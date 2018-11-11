import os
import time
import numpy             as np
import collections       as collections
import pandas            as pd

#import invisible_cities.database.load_db   as db
#import invisible_cities.io.pmaps_io        as pmio

import invisible_cities.database.load_db   as db
from   invisible_cities.io.dst_io          import load_dst
import krcal.dev.corrections               as corrections
import csth .utils.epeak                   as epk
import csth .utils.pmaps_functions         as pmapsf

Q0MIN  = 6.
VDRIFT = 1.

#------------------------------
# City - Driver - CLARICE
#   it read pmaps from hdf5 and produced an event summary Data Frame (corrected energy)
#   if full = True produces also DataFrame for the corrected hits
#------------------------------


def clarice(input_filenames, output_filename, correction_filename, run_number,
            q0min = Q0MIN, full = False):
    """ script to store h5 information per event-peak: position, energy, ...
    if *full* = True it sotres also h5 informacion per slices and hits.
    inputs:
        input_filenames     : (list str) list of pmaps filenames to process
        output_filename     : (str)      name of the output h5 file
        correction_filename : (str)      name of the correction maps
        run_number          : (int)      run number
        q0min               : (float)    minimum charge per SiPM
        full                : (bool)     True writes slice and hits DataFrames, (default = False)
    returns:
        None: appends a h5 outfile a event ('edf') DataFrame with corrected energy, position, ...
        if *full* = True. It also appends a slice ('edf') DataFrame
            and a hit ('hdf') DataFrame with corrected information per slices and hits.
    """

    infiles = [os.path.expandvars(ifile) for ifile in input_filenames]
    corfile = os.path.expandvars(correction_filename)
    outfile = os.path.expandvars(output_filename)

    print('number of input files : ', len(infiles))
    print('correcton file        : ', corfile)
    print('output file           : ', outfile)

    calibrate = corrections.Calibration(corfile, 'scale')

    datasipm = db.DataSiPM(run_number)
    xpos, ypos = datasipm.X.values, datasipm.Y.values

    ntotal, naccepted  = 0, 0

    xtime = np.zeros(len(infiles))
    for i, file in enumerate(infiles):

        xtinit = time.time()
        print('processing ', file)

        try:
            pmaps   = pmapsf.get_pmaps(file, '')
            runinfo = load_dst(file, 'Run', 'events')
        except:
            continue

        partition = epk.partition(file)
        cc = events_summary(pmaps, runinfo, calibrate, xpos, ypos, partition,
                            q0min = q0min, full = full)

        edf = cc[0] if full else cc
        sdf = cc[1] if full else None
        hdf = cc[2] if full else None

        edf .to_hdf(outfile, key = 'edf', append = True)
        if (full):
            sdf.to_hdf(outfile, key = 'sdf', append = True)
            hdf.to_hdf(outfile, key = 'hdf', append = True)

        itot, iacc = len(set(pmaps.s1.event)), len(set(edf.event))
        ntotal += itot; naccepted += iacc
        xtend = time.time()
        xtime[i] = xtend - xtinit

    f = 100.*naccepted /(1.*ntotal) if ntotal > 0 else 0.
    print('total events ', ntotal, ', accepted  ', naccepted, 'fraction (%)' , f)
    print('time per file ', np.mean(xtime), ' s')
    if (naccepted <= 1): naccepted = 1
    print('time per event', np.sum(xtime)/(1.*naccepted), 's')
    return

#---------------------
#   functions per list of pmaps and events
#---------------------

def events_summary(pmaps, runinfo, calibrate, xpos, ypos, loc = 0,
                   q0min = Q0MIN, full = False):
    """ returns a DataFrame with the summary informatio of the event S2 peak: position, energy, ...
    It takes pmaps, and runinfo, calibrate hits, and compute the corrected energy.
    If *full* = True is provided, it returns also a DataFrame with the slices information per event-peak,
    and a DataFrame with the hits information per event-peak
    inputs:
        pmaps     : (DFpmap) pmaps DataFraimes n DFpmap named-tuple
        runinfo   : (DataFrame) runinfo from h5 files
        calibrate : (function) to calibrate the hits
        xpos      : (function) to access x-position of a given SiPM
        ypos      : (function) to access y-poisiton of a given SiPM
        loc       : (int) index of the partition of the file (to acces quickly an event) (default = 0)
        q0min     : (float) minimun charge (pes) per SiPM (default = 6.)
        full      : (bool) if True info per slices and hits are returned
    returns:
        edf : (DataFrame) with the summary informatiom per event-peak: energy, position, size
        if *full* is True:
            sdf : (DataFrame) with the slices information per event-peak: energy, position, ...
            hdf : (DataFrame) with the hits   information per event-peak: energy, position, ...
    """

    spmaps    = pmapsf.filter_1s1(pmaps)

    nepks     = pmapsf.neventpeaks(spmaps)
    etab      = epk.event_table(nepks)

    nslices = len(spmaps.s2)                  if full else 0
    nhits   = np.sum(spmaps.s2i.ene > q0min)  if full else 0
    stab    = epk.slice_table(nslices)        if full else None
    htab    = epk.hit_table  (nhits)          if full else None

    eindex, sindex, hindex = 0, 0, 0
    for iepk, pmap in pmapsf.eventpeak_iterator(spmaps):
        evt, ipk      = iepk

        timestamp     = event_timestamp(runinfo, evt)
        cc            = event_summary(pmap, calibrate, xpos, ypos, q0min = q0min, full = full)
        if (cc is None): continue

        esum = cc[0] if full else cc
        ssum = cc[1] if full else None
        hsum = cc[2] if full else None

        esum.location = loc
        esum.time     = timestamp

        eindex = epk.set_table(esum, eindex, etab)
        if full:
            sindex = epk.set_table(ssum, sindex, stab)
            hindex = epk.set_table(hsum, hindex, htab)

    edf = epk.df_from_table(etab)
    if (not full): return edf

    sdf = epk.df_from_table(stab)
    hdf = epk.df_from_table(htab)
    return edf, sdf, hdf


def event_summary(pmap, calibrate, xpos, ypos,
                  q0min = Q0MIN, full = False):
    """ calibrate hits, compute event informaion: energy, position.
    Returns a Table with the event summary.
    If *full* = True is provided, returns two Tables with the slices and hits information
    inputs:
        pmaps     : (DFpmap) pmaps DataFraimes n DFpmap named-tuple
        calibrate : (function) to calibrate the hits
        xpos      : (function) to access x-position of a given SiPM
        ypos      : (function) to access y-poisiton of a given SiPM
        q0min     : (float) minimun charge (pes) per SiPM (default = 6.)
        full      : (bool) if True info per slices and hits are returned
    returns:
        edf : (Table) with the summary informatiom per event-peak: energy, position, size
        if *full* is True:
            sdf : (Table) with the slices information per event-peak: energy, position, ...
            hdf : (Table) with the hits   information per event-peak: energy, position, ...
    """

    s1, s2, s2i               = pmap

    esum                      = epk.event_table(1)

    evt                       = np.unique(s2.event)[0]
    ipk                       = np.unique(s2.peak)[0]
    esum.event, esum.peak     = evt, ipk

    s1e, t0                   = event_s1_info(s1)
    nslices, z0i, e0i         = event_slices(s2, t0)
    if (nslices <= 0): return None

    nhits, noqhits, q0tot, x0ij, y0ij, z0ij, q0ij = event_hits(s2i, z0i, xpos, ypos, q0min)
    if (nhits <= 0)  : return None

    x0, y0, z0, q0, e0        = epk.eqpoint(e0i, z0i, x0ij, y0ij, q0ij)
    rmax, rsize               = epk.radius(x0ij, y0ij, x0, y0)
    zmin, zmax                = epk.zrange(z0i)

    ceij, cqij                = epk.calibration_factors(x0ij, y0ij, z0ij, calibrate)
    ei, qi, eij, qij          = epk.hits_energy(e0i, z0i, z0ij, q0ij, ceij, cqij)
    ec                        = np.sum(ei)
    ei, enoq, noqslices       = epk.slices_energy(e0i, ei)
    x, y, z, qc, e            = epk.eqpoint(ei , z0i, x0ij, y0ij, qij)

    if (q0 <= 1.): q0 = 1.
    q = qc * q0tot/q0

    esum.s1e, esum.t0         = s1e  , t0
    esum.nslices              = nslices
    esum.nhits, esum.noqhits  = nhits, noqhits
    esum.x0, esum.y0, esum.z0 = x0   , y0, z0
    esum.q0, esum.e0          = q0tot, e0
    esum.rmax, esum.rsize     = rmax , rsize
    esum.zmin, esum.zsize     = zmin , zmax - zmin
    esum.noqslices            = noqslices
    esum.x , esum.y, esum.z   = x    , y, z
    esum.q , esum.e           = q    , e
    esum.qc, esum.ec          = qc   , ec

    if (not full): return esum

    _, q0i, e0ij, _ = epk.hits_energy(e0i, z0i, z0ij, q0ij)

    ssum = epk.slice_table(nslices)
    ssum.event[:], ssum.peak[:] = evt, ipk
    ssum.z0[:]                  = z0i
    ssum.e0[:], ssum.e[:]       = e0i, ei
    ssum.q0[:], ssum.q[:]       = q0i, qi

    hsum = epk.hit_table(nhits)
    hsum.event[:], hsum.peak[:]         = evt, ipk
    hsum.x0[:], hsum.y0[:], hsum.z0[:]  = x0ij, y0ij, z0ij
    hsum.q0[:], hsum.q[:]               = q0ij, qij
    hsum.e0[:], hsum.e[:]               = e0ij, eij

    return esum, ssum, hsum

def event_s1_info(s1):
    """ returns the S1 information: energy and time (t0)
    """

    s1e                  = np.sum(s1.ene)
    if (s1e <= 1.): s1e  = 1.
    t0                   = 1e-3*np.sum(s1.ene*s1.time)/s1e
    #print('s1e ', s1e)
    #print('t0  ', t0)
    #print('time', time)
    return s1e, t0


def event_timestamp(runinfo, evt):
    """ returns the timestamp of an event number *evt* from *runinfo* DataFrame
    """
    timestamp = runinfo[runinfo.evt_number == evt].timestamp.values[0]
    # print(timestamp)
    return timestamp

def event_slices(s2, t0, vdrift = VDRIFT):
    """ return slicaes info: number of slices, z-position and energy of slices
    inputs:
        s2     : (DataFrame) S2 (PMTs) information of an event-peak
        t0     : (float)     t0 value of the S1
        vdrift : (float)     drift velocity (defualt = 1.)
    returns:
        nslices : (int)  number of slices
        z0i     : (array, size = nslices) z-position of the slices
        e0i     : (array, size = nslices) energy of the slices
    """

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
    """ return the hits of the event-peak.
    A Hit is a SiPM in an slice with a charge > *q0min*
    inputs:
        s2i   : (DataFrame) with the S2i (SiPM) information of the event-peak
        z0i   : (array, size = nslices) z-position of the slices of the event-peak
        xpos  : (function) to access x-position of a given SiPM
        ypos  : (function) to access y-position of a given SiPM
        q0min : (float) minimum value (pes) to accept a SiPM
    returns:
        nhits   : (int) number of hits with charge > q0min
        noqhits : (int) number of hits wiht charge < q0min and charge > 0
        q0t0t   : (float) total charge in S2i
        x0ij    : (array, size = nhits) x-position of the hits
        y0ij    : (array, size = nhits) y-position of the hits
        z0ij    : (array, size = nhits) z-position of the hits
        q0ij    : (array, size = nhits) charge of the hits
    """

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
    #selslices    = epk.selection_slices(nslices, nsipms)
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
