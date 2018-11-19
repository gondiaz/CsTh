import numpy             as np
import pandas            as pd

import collections       as collections

import invisible_cities.io.pmaps_io        as pmio
from   invisible_cities.io.dst_io          import load_dst

#----------------------------------------
# Utilities to deal with pmaps-dataframes
#-------------------------------------

PMap = collections.namedtuple('PMap', ['s1', 's2', 's2i'])


def nevents(pmaps):
    """ returns the number of events in pmaps
    """
    nevents = len(np.unique(pmaps.s2.event))
    return nevents


def nepeaks(pmaps):
    """ returns the number of event-peaks in pmaps
    """
    nepks = len(pmaps.s2.groupby(['event', 'peak']))
    return nepks

def events_1s1(pmaps):
    """ returns the list of the events with ony 1S1
    """
    ss1 = pmaps.s1
    evts = np.unique(ss1.groupby('event').filter(lambda x: len(np.unique(x['peak'])) == 1)['event'])
    return evts


def get_event(pmaps, event):
    """ returns the pmap of a given event
    """
    s1, s2, s2i  = pmaps.s1, pmaps.s2, pmaps.s2i
    pm = PMap(s1  [s1 .event == event],
              s2  [s2 .event == event],
              s2i [s2i.event == event])
    return pm


def get_eventpeak(pmaps, event, peak):
    """ returns the pmap of a given event and peak
    """
    s1, s2, s2i  = pmaps.s1, pmaps.s2, pmaps.s2i
    pm = PMap(s1  [ s1 .event == event],
              s2  [(s2 .event == event) & (s2 .peak == peak)],
              s2i [(s2i.event == event) & (s2i.peak == peak)])
    return pm

def event_iterator(pmaps):
    """ returns iterator to iterate along the events of pmaps
    """
    s1, s2, s2i  = pmaps.s1, pmaps.s2, pmaps.s2i

    s1groups     = s1 .groupby('event')
    s2groups     = s2 .groupby('event')
    s2igroups    = s2i.groupby('event')

    for evt, s2item in s2groups:
        s1item  = s1groups .get_group(evt)
        s2iitem = s2igroups.get_group(evt)
        ipmap = PMap(s1item, s2item, s2iitem)
        yield (iepeak, ipmap)


def epeak_iterator(pmaps):
    """ returns iterator to iterate along the event-peaks of pmaps
    """
    s1, s2, s2i  = pmaps.s1, pmaps.s2, pmaps.s2i

    s1groups     = s1 .groupby('event')
    s2groups     = s2 .groupby(['event', 'peak'])
    s2igroups    = s2i.groupby(['event', 'peak'])

    for iepeak, s2item in s2groups:
        evt, ipk = iepeak
        s1item  = s1groups .get_group(evt)
        try:
            s2iitem = s2igroups.get_group(iepeak)
        except:
            continue
        ipmap = PMap(s1item, s2item, s2iitem)
        yield (iepeak, ipmap)

def filter_1s1(pmaps):
    """ filters the pmaps, requirin that the event has only 1 S1
    """
    evts = events_1s1(pmaps)
    tsel = np.isin(pmaps.s1 .event.values, evts)
    ssel = np.isin(pmaps.s2 .event.values, evts)
    hsel = np.isin(pmaps.s2i.event.values, evts)
    return PMap(pmaps.s1[tsel], pmaps.s2[ssel], pmaps.s2i[hsel])


def pmaps_from_hdf(filename):
    """ read the pmaps from a h5 file (official production)
    inputs:
        filename : (str)  the filename of the h5 data
    output:
        pmaps    : (DFPmap) the pmaps (s1, s2, s2i) dataframes
    """
    #try:
    #    hdf = pd.HDFStore(filename)
#   #     dat = [hdf['s1'], hdf['s2'], hdf['s2si'], hdf['s1pmt'], hdf['s2pmt']]
#        dat = (hdf['s1'], hdf['s2'], hdf['s2si'])
#        return DFPmap(*dat)
    #except:
    try:
        s1, s2, s2i, _, _  = pmio.load_pmaps_as_df(filename)
        return PMap(s1, s2, s2i)
    except:
        try:
            hd = pd.HDFStore(filename)
            keys = hd.keys()
            ok = ('/s1' in keys) & ('/s2' in keys) & ('/s2si' in keys)
            if (not ok):
                raise IOError
            s1, s2, s2i = hd['s1'], hd['s2'], hd['s2si']
            return PMap(s1, s2, s2i)
        except:
            raise IOError


def runfo_from_hdf(filename):
    """ read the runinfo from a h5 file
    inputs:
        filename : (str)  the filename of the h5 data
    output:
        pmaps    : (DFPmap) the pmaps (s1, s2, s2i) dataframes
    """
    #try:
    #    hdf = pd.HDFStore(filename)
#   #     dat = [hdf['s1'], hdf['s2'], hdf['s2si'], hdf['s1pmt'], hdf['s2pmt']]
#        dat = (hdf['s1'], hdf['s2'], hdf['s2si'])
#        return DFPmap(*dat)
    #except:
    try:
        runinfo = load_dst(filename, 'Run', 'events')
        return runinfo
    except:
        try:
            hd = pd.HDFStore(filename)
            keys = hd.keys()
            ok = ('/runinfo' in keys)
            if (not ok):
                raise IOError
            runinfo = hd['runinfo']
            return runinfo
        except:
            raise IOError
