from netCDF4 import Dataset

fh=Dataset("DBaseLightConv.nc")

zKu=fh["zKu"][:]
zKa=fh["zKa"][:]
piaKu=fh["piaKu"][:]
piaKuS=fh["piaKuS"][:]
bzd=fh["bzd"][:]
bcf=fh["bcf"][:]
reliab=fh["reliab"][:]
brs=fh["brs"][:]
sfcPrecip=fh["sfcPrecip"][:]
fh.close()
import numpy as np
import time
from datetime import date
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,"/Users/mgrecu/ORO/retr")

import lkTables
from radarRetrSubs import *

piaKuL=[]
sfcPrecipRet=[]
for i,zm1 in enumerate(zKu):
    dnw=np.zeros((176),float)
    bzd1=bzd[i]
    dnw[0:bzd1]=(bzd1-1-np.arange(bzd1))*0.02+0.05
    dnw-=0.2
    bcf1=bcf[i]
    dbin=brs[i]-bcf[i]
    zka_sim,piaKa_sim,kextKaR,asymKaR,salbKaR,\
        kextKaG,salbKaG,asymKaG,\
        zkaG_true,zkaR_true,pRate,dm,piaKu_sim=ret1D(zm1,bzd1,bcf1,alphaS,betaS,alphaR,betaR,dr,lkT,dnw,dbin)
    if sfcPrecip[i]>=0 and pRate[-1]>=0:
        sfcPrecipRet.append([pRate[-1],sfcPrecip[i]])
    
    piaKuL.append(piaKu_sim)
    if i==10000:
        stop
    
