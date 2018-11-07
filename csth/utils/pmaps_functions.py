import numpy             as np
import collections       as collections
import pandas            as pd

import invisible_cities.io.pmaps_io        as pmio

#DFpmap = collections.namedtuple('DFpmaps', ['s1', 's2', 's2i', 's1pmt', 's2pmt'])

DFpmap = collections.namedtuple('DFpmaps', ['s1', 's2', 's2i'])


#----------------------------------------
# Utilities to deal with pmaps-dataframes
#-------------------------------------

def get_pmaps(filename, mode = ''):
    if (mode == 'gd'):
        hdf = pd.HDFStore(filename)
#        dat = [hdf['s1'], hdf['s2'], hdf['s2si'], hdf['s1pmt'], hdf['s2pmt']]
        dat = (hdf['s1'], hdf['s2'], hdf['s2si'])
        return DFpmap(*dat)
    s1, s2, s2i, _, _  = pmio.load_pmaps_as_df(filename)
    return DFpmap(s1, s2, s2i)


def nevents(pmaps):
    nevents = len(np.unique(pmaps.s2.event))
    return nevents


def neventpeaks(pmaps):
    nepks = len(pmaps.s2.groupby(['event', 'peak']))
    return nepks


def events_1s1(pmaps):
    ss1 = pmaps.s1
    evts = np.unique(ss1.groupby('event').filter(lambda x: len(np.unique(x['peak'])) == 1)['event'])
    return evts


def filter_1s1(pmaps):
    evts = events_1s1(pmaps)
    tsel = np.isin(pmaps.s1 .event.values, evts)
    ssel = np.isin(pmaps.s2 .event.values, evts)
    hsel = np.isin(pmaps.s2i.event.values, evts)
    return DFpmap(pmaps.s1[tsel], pmaps.s2[ssel], pmaps.s2i[hsel])


def get_event(pmaps, event):
    s1, s2, s2i  = pmaps
    pm = DFpmap(s1  [s1 .event == event],
                s2  [s2 .event == event],
                s2i [s2i.event == event])
    return pm


def get_eventpeak(pmaps, event, peak):
    s1, s2, s2i  = pmaps
    pm = DFpmap(s1  [ s1 .event == event],
                s2  [(s2 .event == event) & (s2 .peak == peak)],
                s2i [(s2i.event == event) & (s2i.peak == peak)])
    return pm


def event_iterator(pmaps):
    s1, s2, s2i  = pmaps

    s1groups     = s1 .groupby('event')
    s2groups     = s2 .groupby('event')
    s2igroups    = s2i.groupby('event')

    for evt, s2item in s2groups:
        s1item  = s1groups .get_group(evt)
        s2iitem = s2igroups.get_group(evt)
        ipmap = DFpmap(s1item, s2item, s2iitem)
        yield (iepeak, ipmap)


def eventpeak_iterator(pmaps):
    s1, s2, s2i  = pmaps

    s1groups     = s1 .groupby('event')
    s2groups     = s2 .groupby(['event', 'peak'])
    s2igroups    = s2i.groupby(['event', 'peak'])

    for iepeak, s2item in s2groups:
        evt = iepeak[0]
        s1item  = s1groups .get_group(evt)
        s2iitem = s2igroups.get_group(iepeak)
        ipmap = DFpmap(s1item, s2item, s2iitem)
        yield (iepeak, ipmap)
