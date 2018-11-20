import numpy  as np
import pandas as pd


def xy_edges(datasipm, rx, ry):
    '''r is the reduction factor'''
    d=10
    x   , y    =  datasipm['X'], datasipm['Y']
    xmin, xmax =  x.min()      , x.max()
    ymin, ymax =  y.min()      , y.max()

    nx  , ny   = (xmax-xmin)/10 + 1   , (ymax-ymin)/10 + 1
    rg         = [[xmin-d/2, xmax+d/2], [ymin-d/2, ymax+d/2]]

    _, xedges, yedges = np.histogram2d(x, y, bins = [nx, ny], range = rg)

    if (len(xedges)-1)%rx==0 and (len(yedges)-1)%ry==0:
        nx, ny = (len(xedges)-1)/rx, (len(yedges)-1)/ry

        xedges = np.array([xedges[rx*i] for i in range(0, int(nx)+1)])
        yedges = np.array([yedges[ry*i] for i in range(0, int(ny)+1)])

        return xedges, yedges

    else: raise ValueError("reduction factor not allowed")


def xy_histograms(imagedf, xedges, yedges):

    E = imagedf['E']
    emin , emax = E.min(), E.max()
    bined = [xedges, yedges]

    Z = np.sort(np.unique(imagedf['Z']))
    E = []
    for z in Z:
        img = imagedf[imagedf.Z == z]
        Xz, Yz, Ez = np.array(img['X']), np.array(img['Y']), np.array(img['E'])
        H , _ , _  = np.histogram2d(Xz, Yz, weights=Ez, bins=bined)
        H = H.T
        E.append(H)
    return Z, E


def zpartitions(Z, zd):
    '''zd is the desired slice distance'''
    zpart = []
    n = (Z.max()-Z.min())/zd
    i=0
    while True:
        part= [Z[i]]
        for j in range(i+1, len(Z)):
            dj = Z[j]-part[0]
            if dj < zd: part.append(Z[j])
            else: break
        zpart.append(np.array(part))
        if i==j: break
        i = j
    return zpart


def slicemerger(Z, E, zd):
    '''zd is the desired slice distance'''
    zpart = zpartitions(Z, zd)
    lens  = [len(zp) for zp in zpart]

    Em = []
    i, k=0, 0
    while True:
        Esum = E[i]
        for j in range(i+1, i + lens[k]): Esum = Esum + E[j]
        Em.append(Esum)
        i=j
        k+=1
        if k==len(lens):break

    Zm = [np.mean(zp) for zp in zpart]

    return Zm, Em

def imageDataFrame(imagedf, datasipm, rx, ry, zd=-1, th=0):

    xed, yed = xy_edges(datasipm, rx, ry)
    Z  , E   = xy_histograms(imagedf, xed, yed)
    if zd<0: Zm, Em = Z, E
    else:        Zm, Em  = slicemerger(Z, E, zd)

    x = np.array([(xed[i+1] + xed[i])/2. for i in range(0, len(xed)-1)])
    y = np.array([(yed[i+1] + yed[i])/2. for i in range(0, len(yed)-1)])
    x, y =np.meshgrid(x,y)

    X, Y, Z, E = [], [], [], []
    for i in range(0, len(Zm)):
        #thresold data deletion
        Xz, Yz, Ez = x.flatten(), y.flatten(), Em[i].flatten()
        idx0 = []
        for idx in range(0,len(Ez)):
            if Ez[idx]<=th: continue
            else: idx0.append(idx)

        Xz, Yz, Ez = Xz[idx0], Yz[idx0], Ez[idx0]

        X, Y, E = [*X, *list(Xz)], [*Y, *list(Yz)], [*E, *list(Ez)]
        Z       = [*Z, *list(Zm[i]*np.ones(len(Ez)))]

    X, Y, Z, E = np.array(X), np.array(Y), np.array(Z), np.array(E)

    imdf = pd.DataFrame(columns=['X', 'Y', 'Z', 'E'])
    imdf['X'], imdf['Y'], imdf['Z'], imdf['E'] = X, Y, Z, E

    return imdf
