import numpy  as np
import collections as collections
import pandas as pd

#import csth.utils.dfs as dfs
#from   csth.utils.dfs import DF

epeak_vars = [ 'nslices',  # (int)   number of slices
               'nhits'  ,  # (int)   number of hits
               'q0'     ,  # (float) total charge in the epeak
               'e0i'    ,  # (array size=nslices) initial energy per slice
               'zi'     ,  # (array size=nslices) z-position of the slices
               'xij'    ,  # (array size=nhits)   x-position of the hits
               'yij'    ,  # (array size=nhits)   y-position of the hits
               'zij'    ,  # (array size=nhits)   z-position of the hits
               'q0ij']     # (array size=nhits)   initial charge of the hits


EPeak    = collections.namedtuple('EPeak', epeak_vars)

cepeak_vars = epeak_vars + [ 'ei'  , # (array size=nslices)   corrected energy per slice
               'qi'  , # (array size=nslices)   corrected charge per slice
               'eij' , # (array size=nhits)     corrected energy per hit
               'qij' ] # (array size=nhits)     corrected energy per charge

CEPeak = collections.namedtuple('CEPeak', cepeak_vars)



#-------------------------------------------
#   Output
#-------------------------------------------

class ATable:

    def __init__(self, inames, names, size, nints = 0):
        self.inames = inames
        self.names  = names
        self.index  = 0
        dic = {}
        for i, name in enumerate(inames + names):
            dtype    = int if i < nints else float
            dat = np.zeros(size, dtype = dtype)
            setattr(self, name, dat)
            #dic[name] = dat
        #self.df = pd.DataFrame(dic)


    def set(self, obj, loc, size = 1):
        index = self.index
        for iloc, name in zip(loc, self.inames):
            getattr(self, name) [index : index + size] = iloc
        for name in self.names:
            getattr(self, name)[index : index + size] = getattr(obj, name)
            #self.df[name].values[index : index + size] = getattr(obj, name)
        self.index += size
        return


    def __len__(self):
        return self.index



    def __str__(self):
        ss = ''
        for name in self.inames + self.names:
            ss += ' ' + name  + ': ' + str(getattr(self, name))+' \n '
        return ss


    def df(self):
        #return self.df
        dic = {}
        for name in self.inames + self.names : dic[name] = getattr(self, name)
        return pd.DataFrame(dic)


def _clean_df(df):
    df = df[df.event > 0]
    return df

class ESum (ATable):

    inints = 2
    inames = ['event', 'peak']
    enints =  6
    enames = ['location', 'nslices', 'nhits', 'noqslices', 'noqhits', 'time',
              's1e', 't0', 'rmax', 'rsize', 'zmax', 'zsize',
              'x0', 'y0', 'z0', 'e0', 'q0', 'e0h', 'q0h',
              'x' , 'y' , 'z' , 'q' , 'e' , 'eh' , 'qh']

    def __init__(self, size = 1):
        super().__init__(ESum.inames, ESum.enames, size = size,
                         nints = ESum.inints + ESum.enints)

    def to_hdf(self, output_filename):
        df = _clean_df(self.df())
        df.to_hdf(output_filename, key = 'esum', append = True)
        return len(df)

def esum_from_hdf(input_filename):
    hd = pd.HDFStore(input_filename)
    return hd['esum']

class CepkTable:

    inints = 2
    inames = ['event', 'peak']
    enints = 2
    enames = ['nslices', 'nhits', 'q0']
    snames = ['e0i', 'zi', 'ei', 'qi']
    hnames = ['xij', 'yij', 'zij', 'q0ij', 'eij', 'qij']

    def __init__(self, nepks, nslices, nhits):
        self.etab = ATable(CepkTable.inames,  CepkTable.enames, size = nepks,
                           nints = CepkTable.inints + CepkTable.enints)
        self.stab = ATable(CepkTable.inames,  CepkTable.snames, size = nslices,
                           nints = CepkTable.inints)
        self.htab = ATable(CepkTable.inames,  CepkTable.hnames, size = nhits,
                           nints = CepkTable.inints )

    def set(self, cepk, loc):
        #print('cepk size : ', 1, cepk.nslices, cepk.nhits)
        self.etab.set(cepk, loc, size = 1)
        self.stab.set(cepk, loc, size = cepk.nslices)
        self.htab.set(cepk, loc, size = cepk.nhits)

    def df(self):
        edf = _clean_df(self.etab.df())
        sdf = _clean_df(self.stab.df())
        hdf = _clean_df(self.htab.df())
        return (edf, sdf, hdf)


    def to_hdf(self, output_filename):
        df = _clean_df(self.etab.df())
        df.to_hdf(output_filename, key = 'cepk_evt', append = True)
        df = _clean_df(self.stab.df())
        df.to_hdf(output_filename, key = 'cepk_slc', append = True)
        df = _clean_df(self.htab.df())
        df.to_hdf(output_filename, key = 'cepk_hit', append = True)


def cepks_from_hdf(input_filename):
    hd = pd.HDFStore(input_filename)
    edf, sdf, hdf  = hd['cepk_evt'], hd['cepk_slc'], hd['cepk_hit']
    return (edf, sdf, hdf)


#def df_zeros(names, nints, size):
#    dat = {}
#    for i, name in enumerate(names):
#        dtype    = int if i < nints else float
#        dat[name] = np.zeros(size, dtype = dtype)
#    df = pd.DataFrame(dat)
#    return df
#
# inints = 2
# inames = ['event', 'peak']
# enints = 5
# enames = ['location', 'nslices', 'nhits', 'noqslices', 'noqhits', 'time',
#                   's1e', 't0', 'rmax', 'rsize', 'zmax', 'zsize',
#                   'x0', 'y0', 'z0', 'e0', 'q0', 'e0h', 'q0h',
#                   'x' , 'y' , 'z' , 'q' , 'e' , 'eh' , 'qh']
#
# class ESum:
#
#     inints = 2
#     inames = ['event', 'peak']
#     enints = 5
#     enames = ['location', 'nslices', 'nhits', 'noqslices', 'noqhits', 'time',
#               's1e', 't0', 'rmax', 'rsize', 'zmax', 'zsize',
#               'x0', 'y0', 'z0', 'e0', 'q0', 'e0h', 'q0h',
#               'x' , 'y' , 'z' , 'q' , 'e' , 'eh' ,
#
#     def __init__(self, size = 1, names = inames + enames, nints = inints + enints):
#         self.index = 0
#         if (size <= 1):
#             for i in names: setattr(self, name, 0)
#         else:
#             for i, name in enumerate(names):
#                 dtype    = int if i < nints else float
#                 dat = np.zeros(size, dtype = dtype)
#                 setattr(self, name, dat)
#                 #dic[name] = dat
#
#         def set(self, obj, size = 1):
#             index, names = self.index, self.names
#             for name in names:
#                 getattr(self, name)[index : index + size] = getattr(obj, name)
#                 #self.df[name].values[index : index + size] = getattr(obj, name)
#             self.index += size
#             return
#
#
#
# def esum(size = 1, name = 'esum'):
#     at =  ATable(name, inames + enames, size = size, nints = inints + enints)
#     return at
#
#
# def esum_to_hdf(esum, output_filename):
#     #df = esum.df()
#     df = esum.df
#     df = df[df.event > 0]
#     df.to_hdf(output_filename, key = 'esum', append = True)
#     return len(df)
#
# def esum_from_hdf(input_filename):
#
#      hd = pd.HDFStore(input_filename)
#      esum = hd['esum']
#      return esum

#-----------------------------
# Generic hits code
#-----------------------------

def cepeak(epk, calibrate):
    """ create ad corrected-event-peak
    inputs:
        epk    : (EPeak) event peak
    output:
        cepk   : (CEPeak) corrected event peak
    """

    #evt, ipk          = epk.event, epk.peak
    nslices, nhits    = epk.nslices, epk.nhits
    q0                = epk.q0
    e0i, zi           = epk.e0i, epk.zi
    xij, yij, zij     = epk.xij, epk.yij, epk.zij
    q0ij              = epk.q0ij

    # nslices, nhits, q0, e0i, zi, xij, yij, zij, q0ij = epk

    ceij, cqij        = _calibration_factors(xij, yij, zij, calibrate)
    ei, qi, eij, qij  = _calibrate_hits(e0i, zi, zij, q0ij, ceij, cqij)
    ei                = _slices_energy(e0i, ei, qi)

    cepk = CEPeak(nslices, nhits, q0, e0i, zi, xij, yij, zij, q0ij,
                  ei, qi, eij, qij)
    #cepk = (nslices, nhits, q0, e0i, zi, xij, yij, zij, q0ij, ei, qi, eij, qij)
    return cepk


def eqpoint(ei, zi, xij, yij, qij):
    """ compute the average point position and total charge and energy of
    an event-peak.
    inputs:
        ei      : (array, size = nslices) energy per slice:
        zi      : (array, size = nslices) z position of the slices
        xij     : (array, size = nhits)   x position of the hits
        yij     : (array, size = nhits)   y position of the hits
        zij     : (array, size = nhits)   z position of the hits
        qij     : (array, size = nhits) charge of the hits
    returns:
        x, y, z  : (float, float, float) average position
        e, q     : (float, float)        total energy and charge
    """

    ee = np.sum(ei)
    if (ee <= 1.): ee = 1.
    z    = np.sum(zi*ei)/ee

    q = np.sum(qij)
    if (q <= 1.): q = 1.
    x    = np.sum(xij * qij) /q
    y    = np.sum(yij * qij) /q

    #print('x, y, z, q, e ', x0, y0, z0, q0, e0)
    return x, y, z, ee, q


def radius(xij, yij, x, y):
    """ returns the maximum radius respect the origin and the center of the event-peak: (x, y)
    inputs:
        xij  : (array) x position of the hits
        yij  : (array) y position of the hits
        x    : (float) x center
        y    : (float) y center
    returns:
        rmax  : (float) the maximum radius of the hits (repect origin)
        rbase : (float) the maximun radius respect (x0, y0) or base radius
    """
    def _rad(x, y):
        r2 = x*x + y*y
        rmax = np.sqrt(np.max(r2))
        return rmax
    rmax  = _rad( xij    , yij    )
    rbase = _rad( xij - x, yij - y)

    #print('max radius, base radius ', rmax, rbase)
    return rmax, rbase

#----------------------------------------
# Event Summary Table
#-----------------------------------------


def esum(cepk, location, s1e, t0, timestamp):
    """ fill the structure of DataFrames (DFCEPeak) with the information
    of a Corrected Event Peak (CEPeak).
    if full = False (default) returns only event-peak information DF
    if full = True            returns slice and hits DF
    """

    # nslices, nhits, q0, e0i, zi, xij, yij, zij, q0ij, ei, qi, eij, qij = cepk

    esum = ESum(1)

    esum.location = location
    esum.s1e      = s1e
    esum.t0       = t0
    esum.time     = timestamp

    nslices, nhits = cepk.nslices, cepk.nhits
    q0             = cepk.q0
    e0i, zi        = cepk.e0i, cepk.zi
    xij, yij, zij  = cepk.xij, cepk.yij, cepk.zij
    q0ij           = cepk.q0ij
    ei, qi         = cepk.ei, cepk.qi
    eij, qij       = cepk.qi, cepk.eij

    esum.nslices = nslices
    esum.nhits   = nhits

    selnoq          = qi <= 0.
    esum.noqslices  = np.sum(selnoq)

    x0, y0, z0, e0, _       = eqpoint(e0i, zi, xij, yij, q0ij)
    esum.x0, esum.y0, esum.z0  = x0, y0, z0
    esum.e0, esum.q0           = e0, q0
    e0h                     = np.sum(e0i[~selnoq])
    q0h                     = np.sum(q0ij)
    esum.e0h, esum.q0h        = e0h, q0h

    x, y, z, e, _        = eqpoint(ei, zi, xij, yij, qij)
    esum.x, esum.y, esum.z  = x, y, z
    eh                  = np.sum(ei[~selnoq])
    qh                  = np.sum(qi[~selnoq])
    esum.eh, esum.qh     = eh, qh

    fc = qh/q0h if q0h >0 else 0.
    esum.e , esum.q      = e, fc * q0

    rmax, rsize             = radius(xij, yij, x0, y0)
    zmin, zmax              = np.min(zi), np.max(zi)
    esum.rmax, esum.rsize     = rmax, rsize
    esum.zmax, esum.zsize     = zmax, zmax - zmin

    return esum

    # _, q0i, e0ij, _ = _hits(e0i, zi, zij, q0ij)

#-------------------------------------
#   Aunxiliary functions
#--------------------------------------


def _calibration_factors(x, y, z, calibrate):

    nhits = len(z)
    ones = np.ones(nhits)

    #ce0, cq0 = calibrate(x, y, None, None, ones, ones)
    ce , cq  = calibrate(x, y, z   , None, ones, ones)

    #ce0 [ce0 <= 0.] = 1.
    #fe, fq  = ce, cq0*ce/ce0
    fe, fq  = ce, cq

    return fe, fq

#----------------------------------------
#
#-----------------------------------------

def _calibrate_hits(e0i, zi, zij, q0ij, ceij = None, cqij = None):

    nslices = len(zi)
    nhits   = len(zij)

    qij = q0ij * cqij if cqij is not None else q0ij * 1.

    selslices = [zij == izi for izi in zi]
    #selslices = selection_slices_by_z(z0ij, z0i)

    qi  = np.array([np.sum(qij[sel]) for sel in selslices])
    eij = np.zeros(nhits)
    for k, kslice in enumerate(selslices):
        d = 1. if qi[k] <= 1. else qi[k]
        eij[kslice] = qij[kslice] * e0i[k]/qi[k]

    if (ceij is not None): eij = eij * ceij

    ei = np.array([np.sum(eij[sel]) for sel in selslices])

    # noqslices = qi <= 0.
    # e0 = np.sum(e0i[~noqslices])
    # eh = np.sum(ei [~noqslices])
    # fe = eh/e0 if e0 > 0 else 0.
    # ei[noqslices] = fe * e0i [noqslices]

    return ei, qi, eij, qij


def _slices_energy(e0i, ei, qi):

    noqslices = qi <= 0.
    e0 = np.sum(e0i[~noqslices])
    eh = np.sum(ei [~noqslices])
    fe = eh/e0 if e0 > 0 else 0.
    ei[noqslices] = fe * e0i [noqslices]
    return ei
