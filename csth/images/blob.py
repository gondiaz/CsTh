import numpy  as np
import pandas as pd



def blob(imdf, R):
    '''R is the blob radius to consider '''
    Eb = imdf['E'].max()

    Ebdf = imdf[imdf.E==Eb]
    if len(Ebdf)!=1: raise ValueError("two central blob points")

    xb, yb, zb = Ebdf['X'].values, Ebdf['Y'].values, Ebdf['Z'].values

    blobdf  = imdf[((imdf.X-xb)**2 + (imdf.Y-yb)**2 + (imdf.Z-zb)**2) <= R**2]
    imdf_nb = imdf[((imdf.X-xb)**2 + (imdf.Y-yb)**2 + (imdf.Z-zb)**2) >  R**2]

    return blobdf, imdf_nb
