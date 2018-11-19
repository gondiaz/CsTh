import os
import time
import numpy             as np
import collections       as collections
import pandas            as pd

from   invisible_cities.io.dst_io  import load_dst
import krcal.dev.corrections       as corrections
import csth .utils.cepeak          as cpk
from   csth .utils.cepeak          import EPeak, ESum, CepkTable

Q0MIN  = 6.

#------------------------------
# City - Driver - CLARICE
#   it read pmaps from hdf5 and produced an event summary Data Frame (corrected energy)
#   if full = True produces also DataFrame for the corrected hits
#------------------------------



#---------------------------------------------
#  Driver functions to run in a list of files or in a file
#---------------------------------------------


def esum(input_filename, output_filename, correction_filename,
            run_number, location,
            q0min = Q0MIN):

    try:
        hits = data(input_filename)
    except:
        return (0, 0), None

    calibrate = tools(correction_filename, run_number)

    ntotal       = nepeaks(hits)
    esums, cepks = tables(hits, q0min = q0min)

    for iloc, ehits in epeak_iterator(hits):
        evt, ipk = iloc

        epk   = epeak(ehits, q0min)
        if (epk is None): continue

        cepk  = cpk.cepeak(epk, calibrate)
        cepks.set(cepk, iloc)

        timestamp = np.unique(ehits.time)[0]

        esum       = cpk.esum(cepk, location, 0., 0., timestamp)

        esums.set(esum, iloc)

    esums.to_hdf(output_filename)
    cepks.to_hdf(output_filename)

    naccepted = len(esums)
    return (ntotal, naccepted), (esums.df(), cepks.df())


def tools(correction_filename, run_number):

    calibrate  = corrections.Calibration(correction_filename, 'scale')

    return calibrate


def data(input_filename):

    try:
        hits = load_dst(input_filename, 'RECO', 'Events')
    except:
        print('Not able to load file : ', input_filename)
        raise IOError

    print('processing ', input_filename)

    return hits


def tables(hits, q0min = Q0MIN):

    nepks  = nepeaks(hits)
    esums  = cpk.ESum(nepks)

    nslices = len(hits.Q)
    nhits   = np.sum(hits.Q > q0min)
    cepks   = CepkTable(nepks, nslices, nhits)
    return esums, cepks


#-----------------------------
#   Main functions
#-----------------------------

def nepeaks(hits):

    groups = hits.groupby(['event', 'npeak'])
    return len(groups)


def epeak_iterator(hits):

    groups = hits.groupby(['event', 'npeak'])
    return iter(groups)


def epeak(hits, q0min = Q0MIN):
    """ returns information form the event-peak (per slices and hits)
    inputs:
        pmap  : (DFPmap)    pmap info per a given event-peak
        q0min : (float)
    returns:
        q0    : (float) total charge in the event-peak
        n0hits: (int)   total number of SiPMs with a signal > 0
        e0i   : (np.array, size = nslices) energy of the slices
        zi    : (np.array, size = nslices) z-position of the slices
        xij, yij, zij : (np.array, size = nhits) x, y, z posisition of the SiPMs with charge > q0min
        q0ij  : (np.array, size = nhits) charge of the SiPMs with charge > q0min
    """

    q0                         = np.sum(hits.Q [hits.Q > 0])

    nslices, e0i, zi           = _slices(hits)
    if (nslices < 0): return None

    nhits, xij, yij, zij, q0ij = _hits(hits, q0min)
    if (nhits <= 0):  return None

    return EPeak(nslices, nhits, q0, e0i, zi, xij, yij, zij, q0ij)


#---------------------------


def _slices(hits):

    zij    = hits.Z.values
    zi     = np.unique(zij)
    nslices = len(zi)
    if (nslices <= 0):
        return nslices, None, None

    e0ij      = hits.E.values
    selslices = [zij == izi for izi in zi]
    e0i  = np.array([np.sum(e0ij[sel]) for sel in selslices])

    #print('nslices, t0, e0i, zi', nslices, t0, e0i, zi)
    return nslices, e0i, zi


def _hits(hits, q0min = Q0MIN):

    qsel    = hits.Q > q0min
    nhits   = np.sum(qsel)
    if (nhits <= 0):
            return nhits, None, None, None, None

    q0ij   = hits.Q [qsel].values
    xij    = hits.X [qsel].values
    yij    = hits.Y [qsel].values
    zij    = hits.Z [qsel].values

    #print('nhits, xij, yij, zij, q0ij ', nhits, len(xij), len(yij), len(zij), len(q0ij))
    #print('xij, yij, zij, q0ij ', xij, yij, zij, q0ij)
    return nhits, xij, yij, zij, q0ij



#---------------------###############
#   functions per list of pmaps and events
#---------------------
#
#
# def events_summary(hits, calibrate, loc = 0, q0min = Q0MIN, full = False):
#     """ returns a DataFrame with the summary information of event-peaks: position, energy, ...
#     It takes hits from hdsts, calibrate them, and compute the corrected energy.
#     If *full* = True is provided, it returns also a DataFrame with the slices information per event-peak,
#     and a DataFrame with the hits information per event-peak
#     inputs:
#         hits      : (DataFrame) hits information (hdst DataFrame)
#         calibrate : (function) to calibrate hits
#         loc       : (int) index of the partition of the file (to acces quickly an event) (default = 0)
#         q0min     : (float) minimun charge (pes) per SiPM (default = 6.)
#         full      : (bool) if True info per slices and hits are returned
#     returns:
#         edf : (DataFrame) with the summary informatiom per event-peak: energy, position, size
#         if *full* is True:
#             sdf : (DataFrame) with the slices information per event-peak: energy, position, ...
#             hdf : (DataFrame) with the hits   information per event-peak: energy, position, ...
#     """
#
#     groups = hits.groupby(['event', 'npeak'])
#
#     nepks  = len(groups)
#     etab = epk.event_table(nepks)
#
#     nslices = len(hits)                  if full else 0 # TODO: compute it properly!
#     nhits   = len(hits)                  if full else 0
#     stab    = epk.slice_table(nslices)   if full else None
#     htab    = epk.hit_table  (nhits)     if full else None
#
#     eindex, sindex, hindex = 0, 0, 0
#     for iepk, ehits  in groups:
#         esum = event_summary(ehits, calibrate, loc, q0min)
#
#         cc            = event_summary(ehits, calibrate, q0min, full)
#         if (cc is None): continue
#
#         esum = cc[0] if full else cc
#         ssum = cc[1] if full else None
#         hsum = cc[2] if full else None
#
#         esum.location = loc
#
#         eindex = epk.set_table(esum, eindex, etab)
#         if full:
#             sindex = epk.set_table(ssum, sindex, stab)
#             hindex = epk.set_table(hsum, hindex, htab)
#
#     edf = epk.df_from_table(etab)
#     if (not full): return edf[edf.event > 0]
#
#     sdf = epk.df_from_table(stab)
#     hdf = epk.df_from_table(htab)
#     return edf[edf.event > 0], sdf[sdf.event > 0], hdf[hdf.event >0]
#
#
# def event_summary(hits, calibrate, q0min = Q0MIN, full = False):
#     """ returns a DataFrame with the summary informatio of an event-peak: position, energy, ...
#     It takes hits from hdsts, calibrate them, and compute the corrected energy.
#     If *full* = True is provided, it returns also a DataFrame with the slices information per event-peak,
#     and a DataFrame with the hits information per event-peak
#     inputs:
#         hits      : (DataFrame) hits information for an event-peak
#         calibrate : (function)  to calibrate hits
#         q0min     : (float) minimun charge (pes) per SiPM (default)
#         full      : (bool) if True info per slices and hits are returned
#     returns:
#         edf : (Table) with the summary informatiom per event-peak: energy, position, size
#         if *full* is True:
#             sdf : (Table) with the slices information per event-peak: energy, position, ...
#             hdf : (Table) with the hits   information per event-peak: energy, position, ...
#     """
#
#     esum                      = epk.event_table(1)
#
#     evt                       = np.unique(hits.event)[0]
#     ipk                       = np.unique(hits.npeak)[0]
#     esum.event, esum.peak     = evt, ipk
#
#     s1e, t0, timestamp        = event_s1_info(hits)
#     nslices, z0i, e0i         = event_slices(hits)
#     if (nslices <= 0): return None
#
#     nhits, noqhits, q0tot, x0ij, y0ij, z0ij, q0ij = event_hits(hits, z0i, q0min)
#     if (nhits <= 0):   return None
#
#     x0, y0, z0, q0, e0        = epk.eqpoint(e0i, z0i, x0ij, y0ij, q0ij)
#     rmax, rsize               = epk.radius(x0ij, y0ij, x0, y0)
#     zmin, zmax                = epk.zrange(z0i)
#
#     ceij, cqij                = epk.calibration_factors(x0ij, y0ij, z0ij, calibrate)
#     ei, qi, eij, qij          = epk.hits_energy(e0i, z0i, z0ij, q0ij, ceij, cqij)
#     ec                        = np.sum(ei)
#     ei, enoq, noqslices       = epk.slices_energy(e0i, ei)
#     x, y, z, qc, e            = epk.eqpoint(ei , z0i, x0ij, y0ij, qij)
#
#     if (q0 <= 1.): q0 = 1.
#     q = qc * q0tot/q0
#
#     esum.s1e, esum.t0         = s1e  , t0
#     esum.nslices              = nslices
#     esum.nhits, esum.noqhits  = nhits, noqhits
#     esum.x0, esum.y0, esum.z0 = x0   , y0, z0
#     esum.q0, esum.e0          = q0tot, e0
#     esum.rmax, esum.rsize     = rmax , rsize
#     esum.zmin, esum.zsize     = zmin , zmax - zmin
#     esum.noqslices            = noqslices
#     esum.x , esum.y, esum.z   = x    , y, z
#     esum.q , esum.e           = q    , e
#     esum.qc, esum.ec          = qc   , ec
#
#     if (not full): return esum
#
#     _, q0i, e0ij, _ = epk.hits_energy(e0i, z0i, z0ij, q0ij)
#
#     ssum = epk.slice_table(nslices)
#     ssum.event[:], ssum.peak[:] = evt, ipk
#     ssum.z0[:]                  = z0i
#     ssum.e0[:], ssum.e[:]       = e0i, ei
#     ssum.q0[:], ssum.q[:]       = q0i, qi
#
#     hsum = epk.hit_table(nhits)
#     hsum.event[:], hsum.peak[:]         = evt, ipk
#     hsum.x0[:], hsum.y0[:], hsum.z0[:]  = x0ij, y0ij, z0ij
#     hsum.q0[:], hsum.q[:]               = q0ij, qij
#     hsum.e0[:], hsum.e[:]               = e0ij, eij
#
#     return esum, ssum, hsum
#
#
# def event_s1_info(hits):
#     """ returns the s1, time-0 and timestamp of the event
#     """
#     s1e  = 0.
#     t0   = 0.
#     time = np.unique(hits.time)[0]
#
#     #print('s1e , t0, time ', s1e, t0, time)
#     return 0., 0., time
#
#
# def event_slices(hits):
#     """ returns the number of slices, z-position and energy of the slices
#     """
#
#     z0ij    = hits.Z.values
#     z0i     = np.unique(z0ij)
#     nslices = len(z0i)
#     if (nslices <= 0):
#         return nslices, None, None
#
#     e0ij        = hits.E.values
#     selslices = epk.selection_slices_by_z(z0ij, z0i)
#     e0i  = np.array([np.sum(e0ij[sel]) for sel in selslices])
#
#     #print('nslices ', nslices)
#     #print('e0i ', len(e0i) , np.sum(e0i)  , e0i)
#     #print('z0i ', len(z0i),  np.sum(z0i*e0i)/np.sum(e0i), z0i)
#     return nslices, z0i, e0i
#
#
# def event_hits(hits, z0i, q0min = Q0MIN):
#     """ return the number of hits, position and charge of the hits
#     """
#
#     ntot  = len(hits.Q)
#     qsel  = hits.Q > q0min
#     q0tot = np.sum(hits.Q[hits.Q > 0.])
#     nhits = int(np.sum(qsel))
#     noqhits = ntot - nhits
#
#     if (nhits <= 0):
#         return nhits, noqhits, q0tot, None, None, None, None
#
#     q0ij   = hits.Q[qsel].values
#     x0ij   = hits.X[qsel].values
#     y0ij   = hits.Y[qsel].values
#     z0ij   = hits.Z[qsel].values
#
#     #print('nhits ', nhits, 'noqhits', noqhits, 'q0tot', q0tot)
#     #print('x0ij', len(x0ij), x0ij, '\n y0ij', len(y0ij), y0ij, '\n z0ij', len(z0ij), z0ij)
#     #print('q0ij', len(q0ij), q0ij)
#     return nhits, noqhits, q0tot, x0ij, y0ij, z0ij, q0ij
#
# #----------------------
