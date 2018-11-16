import numpy             as np
import pandas            as pd

import invisible_cities.io.pmaps_io        as pmio

#----------------------------------------
# Utilities to deal with pmaps-dataframes
#-------------------------------------


class DFPmap:
    """ pmaps container: (s1, s2 (PMT), s2i (SiPM)) information
    """

    def __init__(self, s1, s2, s2i):
        self.s1  = s1
        self.s2  = s2
        self.s2i = s2i

    def nevents(self):
        """ returns the number of events in pmaps
        """
        nevents = len(np.unique(self.s2.event))
        return nevents


    def nepeaks(self):
        """ returns the number of event-peaks in pmaps
        """
        nepks = len(self.s2.groupby(['event', 'peak']))
        return nepks

    def events_1s1(self):
        """ returns the list of the events with ony 1S1
        """
        ss1 = self.s1
        evts = np.unique(ss1.groupby('event').filter(lambda x: len(np.unique(x['peak'])) == 1)['event'])
        return evts


    def get_event(self, event):
        """ returns the pmap of a given event
        """
        s1, s2, s2i  = self.s1, self.s2, self.s2i
        pm = DFPmap(s1  [s1 .event == event],
                    s2  [s2 .event == event],
                    s2i [s2i.event == event])
        return pm


    def get_eventpeak(self, event, peak):
        """ returns the pmap of a given event and peak
        """
        s1, s2, s2i  = self.s1, self.s2, self.s2i
        pm = DFPmap(s1  [ s1 .event == event],
                    s2  [(s2 .event == event) & (s2 .peak == peak)],
                    s2i [(s2i.event == event) & (s2i.peak == peak)])
        return pm

    def event_iterator(self):
        """ returns iterator to iterate along the events of pmaps
        """
        s1, s2, s2i  = self.s1, self.s2, self.s2i

        s1groups     = s1 .groupby('event')
        s2groups     = s2 .groupby('event')
        s2igroups    = s2i.groupby('event')

        for evt, s2item in s2groups:
            s1item  = s1groups .get_group(evt)
            s2iitem = s2igroups.get_group(evt)
            ipmap = DFPmap(s1item, s2item, s2iitem)
            yield (iepeak, ipmap)


    def epeak_iterator(self):
        """ returns iterator to iterate along the event-peaks of pmaps
        """
        s1, s2, s2i  = self.s1, self.s2, self.s2i

        s1groups     = s1 .groupby('event')
        s2groups     = s2 .groupby(['event', 'peak'])
        s2igroups    = s2i.groupby(['event', 'peak'])

        for iepeak, s2item in s2groups:
            evt = iepeak[0]
            s1item  = s1groups .get_group(evt)
            try:
                s2iitem = s2igroups.get_group(iepeak)
            except:
                continue
            ipmap = DFPmap(s1item, s2item, s2iitem)
            yield (iepeak, ipmap)

    def to_hdf(self, output_filename):
        """ store the pmaps into an hdf5 file
        """

        s1, s2, s2i  = pmaps.s1, pmaps.s2, pmaps.s2i

        s1.to_hdf(output_filename, key = 's1'  , append = True)
        s1.to_hdf(output_filename, key = 's2'  , append = True)
        s1.to_hdf(output_filename, key = 's2si', append = True)


def filter_1s1(pmaps):
    """ filters the pmaps, requirin that the event has only 1 S1
    """
    evts = pmaps.events_1s1()
    tsel = np.isin(pmaps.s1 .event.values, evts)
    ssel = np.isin(pmaps.s2 .event.values, evts)
    hsel = np.isin(pmaps.s2i.event.values, evts)
    return DFPmap(pmaps.s1[tsel], pmaps.s2[ssel], pmaps.s2i[hsel])


def dfpmaps_from_hdf(filename):
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
        return DFPmap(s1, s2, s2i)
    except:
        print('Not able to load pmaps from file ', filename)
        raise IOError
