import time
import numpy             as np
import pandas            as pd

import invisible_cities.database.load_db   as db
from   invisible_cities.io.dst_io          import load_dst
import krcal.dev.corrections               as corrections

import csth .utils.cepeak           as cpk
from   csth .utils.cepeak           import EPeak, CEPeak, ESum, CepkTable
import csth .utils.dfpmaps          as pmapdf

Q0MIN  = 6.
VDRIFT = 1.

#---------------------------------------------
#  Driver functions to run in a list of files or in a file
#---------------------------------------------


def esum(input_filename, output_filename, correction_filename,
            run_number, location,
            q0min = Q0MIN):

    try:
        pmaps, runinfo = data(input_filename)
    except:
        return (0, 0), None

    calibrate, xpos, ypos = tools(correction_filename, run_number)

    ntotal      = pmaps.nepeaks()
    esums, cepks = tables(pmaps, q0min = q0min)

    for iloc, pmap in pmaps.epeak_iterator():
        evt, ipk = iloc

        epk   = epeak(pmap, xpos, ypos, q0min)
        if (epk is None): continue

        cepk  = cpk.cepeak(epk, calibrate)
        cepks.set(cepk, iloc)

        s1e, t0    = s1_info(pmap.s1)
        timestamp  = 0 if runinfo is None else runinfo[runinfo.evt_number == evt].timestamp.values[0]
        esum       = cpk.esum(cepk, location, s1e, t0, timestamp)

        esums.set(esum, iloc)

    esums.to_hdf(output_filename)
    cepks.to_hdf(output_filename)

    naccepted = len(esums)
    return (ntotal, naccepted), (esums.df(), cepks.df())


def tools(correction_filename, run_number):

    calibrate  = corrections.Calibration(correction_filename, 'scale')
    datasipm   = db.DataSiPM(run_number)
    xpos, ypos = datasipm.X.values, datasipm.Y.values

    return calibrate, xpos, ypos

def data(input_filename):

    try:
        pmaps   = pmapdf.dfpmaps_from_hdf(input_filename)
        runinfo =        load_dst(input_filename, 'Run', 'events')
    except:
        print('Not able to load file : ', input_filename)
        raise IOError

    print('processing ', input_filename)

    spmaps      = pmapdf.filter_1s1(pmaps)

    return spmaps, runinfo

def tables(pmaps, q0min = Q0MIN):

    nepks  = pmaps.nepeaks()
    esums  = cpk.ESum(nepks)

    nhits = np.sum(pmaps.s2i.ene > q0min)
    cepks = CepkTable(nepks, nhits, nhits)
    return esums, cepks


#-----------------------------
#   Main functions
#-----------------------------


def epeak(pmap, xpos, ypos, q0min = Q0MIN):
    """ returns information form the event-peak (per slices and hits)
    inputs:
        pmap  : (DFPmap)    pmap info per a given event-peak
        xpos  : (function)
        ypos  : (function)
        q0min : (float)
    returns:
        epk   : (EPeak) evenr-peak (only raw information)
    """

    s1, s2, s2i                = pmap.s1, pmap.s2, pmap.s2i

    #evt = np.unique(s2.event)[0]
    #ipk = np.unique(s2.peak) [0]

    q0                         = np.sum(s2i.ene)

    nslices, t0, e0i, zi       = _slices(s1, s2)
    if (nslices <= 0): return None

    nhits, xij, yij, zij, q0ij = _hits(zi, s2i, xpos, ypos, q0min)
    if (nhits <= 0):  return None

    epk = EPeak(nslices, nhits, q0, e0i, zi, xij, yij, zij, q0ij)

    #epk = EPeak(evt, ipk, nslices, nhits, q0, e0i, zi, xij, yij, zij, q0ij)
    #epk = (nslices, nhits, q0, e0i, zi, xij, yij, zij, q0ij)
    return epk


def s1_info(s1):

    s1e                  = np.sum(s1.ene)
    if (s1e <= 1.): s1e  = 1.
    t0                   = 1e-3*np.sum(s1.ene*s1.time)/s1e

    return s1e, t0

#-------------------------
#   Auxiliary functions
#-------------------------

def cepks_init(pmaps, q0):
    nepks = pmaps.nepeaks()
    nhits  = np.sum(pmaps.s2i.ene > q0)
    cepks = CepksTable(nepks, nhits, nhits)
    return cepks

def _slices(s1, s2):

    s1e                  = np.sum(s1.ene)
    if (s1e <= 1.): s1e  = 1.
    t0                   = 1e-3*np.sum(s1.ene*s1.time)/s1e

    ts  = 1.e-3*s2.time.values
    nslices = len(ts)
    if (nslices <= 0):
        return nslices, t0, None, None
    zi  = VDRIFT*(ts-t0)
    e0i  = s2.ene.values

    #print('nslices, t0, e0i, zi', nslices, t0, e0i, zi)
    return nslices, t0, e0i, zi


def _hits(zi, s2i, xpos, ypos, q0min = Q0MIN):

    nslices      = len(zi)

    q0ij         = s2i.ene.values
    ntotal_hits  = len(q0ij)
    nsipms       = int(ntotal_hits/nslices)
    assert int(nsipms*nslices) == ntotal_hits

    qtot  = np.sum(q0ij)
    zij = np.tile(zi, nsipms)

    qsel    = q0ij > q0min
    noqsel  = (q0ij > 0) & (q0ij <= q0min)
    nhits   = np.sum(qsel)
    if (nhits <= 0):
            return nhits, None, None, None, None

    sipm   = s2i.nsipm.values
    q0ij   = q0ij[qsel]
    xij    = xpos[sipm[qsel]]
    yij    = ypos[sipm[qsel]]
    zij    = zij [qsel]

    #print('nhits, xij, yij, zij, q0ij ', nhits, len(xij), len(yij), len(zij), len(q0ij))
    #print('xij, yij, zij, q0ij ', xij, yij, zij, q0ij)

    return nhits, xij, yij, zij, q0ij
