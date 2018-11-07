import numpy             as np
import pandas            as pd


def imageDataFrame(s1_ev, s2_ev, s2si_ev, datasipm, VDRIFT=1.):
    '''Given s2 dataframe and s2si dataframe for a given event and peak and
     sipm data it returns the dataframe with s2si_ev + sipm positions + time'''
    df     = _times_to_s2si  (s2_ev, s2si_ev)          #this add times to s2si
    si_pos = _sipmpos_to_s2si(s2_ev, s2si_ev, datasipm)#sipm pos to append
    imagedf = pd.concat([df, si_pos], axis=1)

    #ADDING Z positions from time
    t0 = float(s1_ev['time'].head(1))
    Z  = (imagedf['time'] - t0)*VDRIFT*1.e-3
    imagedf = imagedf.assign(Z=Z)

    imagedf = imagedf.rename(columns = {'ene':'E'})
    return imagedf


def _times_to_s2si(s2_ev, s2si_ev):
    '''Given s2 dataframe and s2si dataframe for a given event and peak,
    it returns the s2si dataframe with time info'''
    times   = np.array(s2_ev['time'])
    n_times = len(times)

    df1   = s2si_ev                 .reset_index(drop=True)
    df2_1 = s2_ev['time'].to_frame().reset_index(drop=True)

    df2   = pd.concat([df2_1, df2_1], ignore_index=True)
    while len(df2)<len(df1):
        df2 = pd.concat([df2, df2_1], ignore_index=True)
    df  = pd.concat([df1, df2]  , axis = 1)

    return df


def _sipmpos_to_s2si(s2_ev, s2si_ev, datasipm):
    '''Given s2 dataframe and s2si dataframe for a given event and peak and
     sipm data it returns the dataframe with sipm positions to be added to s2si'''
    df1 = s2si_ev.reset_index(drop=True)
    n_times = len(s2_ev['time'])

    si_pos = pd.DataFrame(columns = ['X', 'Y'])
    sipms = df1.loc[[n_times*i for i in range(0,int(len(df1)/n_times))]]['nsipm']
    X, Y = pd.Series([], name='X'), pd.Series([], name='Y')

    for sipm in sipms:
        si_info = datasipm.loc[sipm][['X','Y']]

        pos = np.array(si_info)*np.ones([n_times, 2])

        X = X.append(pd.Series(pos[:,0]), ignore_index = True)
        Y = Y.append(pd.Series(pos[:,1]), ignore_index = True)

    si_pos['X'], si_pos['Y'] = X, Y
    return si_pos
