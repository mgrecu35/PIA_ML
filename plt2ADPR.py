#!/usr/bin/env python
import numpy as np
import time
from datetime import date
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,"/Users/mgrecu/ORO/retr")

import lkTables
from radarRetrSubs import *


import pickle
dateL=[]


from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
import glob
fs=sorted(glob.glob("2A-CS/2A*"))

from numpy import *
def readOrb(orb):
    fh=Dataset(orb)
    Lon=fh['FS/Longitude'][:,24]
    Lat=fh['FS/Latitude'][:,24]
    a=nonzero((Lon+75)*(Lon+105)<0)
    b=nonzero((Lat[a]-30)*(Lat[a]-50)<0)
    sfcPrecip=fh['FS/SLV/precipRateNearSurface'][:,:]
    lon=fh['FS/Longitude'][:,:]
    lat=fh['FS/Latitude'][:,:]
    zsfc=fh['FS/PRE/elevation'][:,:]
    hzero=fh['FS/VER/heightZeroDeg'][:,:]
    a1=nonzero(zsfc>300)
    bcf=fh['FS/PRE/binClutterFreeBottom'][:,:]
    brs=fh['FS/PRE/binRealSurface'][:,:]
    bst=fh['FS/PRE/binStormTop'][:,:]
    bzd=fh['FS/VER/binZeroDeg'][:,:]
    zku=fh['FS/PRE/zFactorMeasured'][:,:,:]
    t=fh['FS/ScanTime/Hour'][:]+fh['FS/ScanTime/Minute'][:]/60.0
    pType=fh['FS/CSF/typePrecip'][:,:]
    #pType=(pType/1e7).astype(int)
    zkuc=fh['FS/SLV/zFactorFinal'][:,:,:]
    return zku,bzd,bst,brs,bcf,zsfc,hzero,lon,lat,fh,t,sfcPrecip,pType,zkuc
eRad=6357e3


orbL=sorted(glob.glob("2A-CS/*HDF5"))

d=pickle.load(open("orbitList.pklz","rb"))
fList=d["fList"]
orbList=[".029945.",".029521.",".029490.",".029951.",".030052.",".030099.",".030581.",".030760."]
rg=[[100,350],[550,650],[100,400],[350,450],[240,500],[250,450],[110,240],[400,650]]
import bisectm
nwdm=np.loadtxt("NwDm.txt")

for orb in fList[:]:
    if orb[0][62:70] in orbList[:1]:
        zku,bzd,bst,brs,bcf,zsfc,hzero,lon,lat,fh,t,sfcPrecip,pType,zkuc=\
            readOrb(orb[0])
        pType=(pType/1e7).astype(int)
        if sfcPrecip.sum()>10000:
            plt.subplot(211)
            plt.pcolormesh(zku[:,24,::-1,0].T,cmap='jet',vmin=0,vmax=48)
            plt.title(orb[0][38:-6])
            ind=orbList.index(orb[0][62:70])
            plt.xlim(rg[ind][0],rg[ind][1])
            s0=fh['FS/PRE/sigmaZeroMeasured'][:]
            j=24
            nsfcRate=[]
            pRateL=[]
            bbPeak=fh['FS/CSF/binBBPeak'][:]
            dmdnwL=[]
            for i in range(rg[ind][0],rg[ind][1]):
                dnw=np.zeros((176),float)
                zm1=zku[i,j,:,0]
                bzd1=bzd[i,j]
                dnw[0:bzd1]=(bzd1-1-np.arange(bzd1))*0.02+0.05
                dnw+=0.3
                bcf1=bcf[i,j]
                zc1=zm1.copy()
                if bbPeak[i,j]<=0:
                    zka_sim,pia_sim2,kextKaR,asymKaR,salbKaR,\
                        kextKaG,salbKaG,asymKaG,\
                        zkaG_true,zkaR_true,pRate,dm=ret1D(zm1,bzd1,bcf1,alphaS,betaS,alphaR,betaR,dr,lkT,dnw)
                else:
                    zka_sim,pia_sim2,kextKaR,asymKaR,salbKaR,\
                        kextKaG,salbKaG,asymKaG,\
                        zkaG_true,zkaR_true,pRate,dm=ret1Dst(zm1,bzd1,bbPeak[i,j],bcf1,alphaS,betaS,alphaR,\
                                                          betaR,dr,lkT,dnw)
                    
                pRate[pRate<0]=0
                if pRate[-1]>0:
                    ibin=bisectm.bisectm(nwdm[:,0],60,dm[-1])
                    nw1=nwdm[ibin,1]-np.log10(0.08e8)
                    dnw+=0.5*(nw1-dnw)
                    zc1=zm1.copy()
                    if bbPeak[i,j]<=0:
                        zka_sim,pia_sim2,kextKaR,asymKaR,salbKaR,\
                            kextKaG,salbKaG,asymKaG,\
                            zkaG_true,zkaR_true,pRate,dm=ret1D(zm1,bzd1,bcf1,alphaS,betaS,alphaR,betaR,dr,lkT,dnw)
                    else:
                        zka_sim,pia_sim2,kextKaR,asymKaR,salbKaR,\
                            kextKaG,salbKaG,asymKaG,\
                            zkaG_true,zkaR_true,pRate,dm=ret1Dst(zm1,bzd1,bbPeak[i,j],bcf1,alphaS,betaS,alphaR,\
                                                                 betaR,dr,lkT,dnw)
                nsfcRate.append(pRate[-1])
                if pType[i,j]==1:
                    dmdnwL.append([dm[-1],dnw[-1]])
            plt.subplot(212)
            plt.plot(nsfcRate)
            plt.plot(sfcPrecip[rg[ind][0]:rg[ind][1],j])

            plt.show()
            
            #fout='FigsL/'+orb[0][38:-6]+'.png'
            #plt.savefig(fout)





plt.scatter(nwdm[:,0],nwdm[:,1])
