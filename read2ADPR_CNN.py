#!/usr/bin/env python
#https://www.youtube.com/watch?v=ilkSwsggSNM
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
fs=sorted(glob.glob("2A-CS/2A*KWAJ*"))

from numpy import *
def readOrb(orb):
    fh=Dataset(orb)
    sfcPrecip=fh['FS/SLV/precipRateNearSurface'][:,:]
    precipRate=fh['FS/SLV/precipRate'][:,:]
    lon=fh['FS/Longitude'][:,:]
    lat=fh['FS/Latitude'][:,:]
    hzero=fh['FS/VER/heightZeroDeg'][:,:]
    pType=fh['FS/CSF/typePrecip'][:,:]
    stormTop=fh['FS/PRE/heightStormTop'][:,:]
    pType=(pType/1e7).astype(np.int)
    bzd=fh['FS/VER/binZeroDeg'][:,:]
    zku=fh['FS/PRE/zFactorMeasured'][:,:,:,0]
    bcf=fh['FS/PRE/binClutterFreeBottom'][:,:]
    dm=fh['FS/SLV/paramDSD'][:,:,:,1]
    zkuc=fh['FS/SLV/zFactorFinal'][:,:,:,0]
    piaF=fh['FS/SLV/piaFinal'][:,:,:]
    return sfcPrecip,hzero,pType,stormTop,bzd,zku,bcf,precipRate,dm,zkuc,piaF,fh
eRad=6357e3


orbL=sorted(glob.glob("2A-CS/*KWAJ*HDF5"))
pTypeL=[]
nt=0
sfcPrecipL=[]
stormTopL=[]
zKuFL1=[]
zKuFL2=[]
zKuL=[]
dbinL=[]
sfcPrecip1L=[]
pRateL=[]
pTypeL=[]
n1=4
sfcPrecip2L=[]
pTypeL=[]
dmL=[]
precipRateL=[]
stormTop2L=[]
piaF2L=[]
zcL=[]
for orb in fs[:300]:
    try:
        sfcPrecip,hzero,pType,stormTop,bzd,zku,bcf,precipRate,dm,zkuc,piaF,fh=readOrb(orb)
    except:
        pass
    stormTop.data[stormTop.data<0]=0
    a=np.nonzero(pType>0)
    b=np.nonzero(hzero[a]>3000)
    #nt+=len(a[0][b])
    nx=sfcPrecip.shape[0]
    for i1,j1 in zip(a[0][b],a[1][b]):
        if i1>5 and i1<nx-5 and j1>12 and j1<37:
            if sfcPrecip[i1-5:i1+6,j1].data.min()>-1 and \
               sfcPrecip[i1-1:i1+2,j1].data.max()>0:
                zc1L=np.zeros((2*n1+1,2*n1+1),float)
                dm1L=np.zeros((2*n1+1,2*n1+1),float)
                pRate1L=np.zeros((2*n1+1,2*n1+1),float)
                dbinL.append(bcf[i1,j1]-bzd[i1,j1])
                if (bcf[i1-n1:i1+n1+1,j1-n1:j1+n1+1]-bzd[i1-n1:i1+n1+1,j1-n1:j1+n1+1]).min()>15:
                    zsect=[zku[i1+k1,j1+k2,bzd[i1+k1,j1+k2]-25:bzd[i1+k1,j1+k2]+15] for k1 in range(-n1,n1+1) for k2 in range(-n1,n1+1)]
                    for k1 in range(-n1,n1+1):
                        for k2 in range(-n1,n1+1):
                            zc1L[k1+n1,k2+n1]=zkuc[i1+k1,j1+k2,bzd[i1+k1,j1+k2]+15]
                            dm1L[k1+n1,k2+n1]=dm[i1+k1,j1+k2,bzd[i1+k1,j1+k2]+15]
                            pRate1L[k1+n1,k2+n1]=precipRate[i1+k1,j1+k2,bzd[i1+k1,j1+k2]+15]
                    pratem=pRate1L.mean()
                    dm1L[dm1L!=dm1L]=0
                    zc1L[zc1L!=zc1L]=0
                    if pratem!=pratem:
                        continue
                    zcL.append(zc1L)
                    dmL.append(dm1L)
                    precipRateL.append(pRate1L)
                    stormTop2L.append(stormTop[i1-n1:i1+n1+1,j1-n1:j1+n1+1])
                    piaF2L.append(piaF[i1-n1:i1+n1+1,j1-n1:j1+n1+1])
                    zKuL.append(np.array(zsect).T)
                    pRateL.append(precipRate[i1,j1,bzd[i1,j1]-25:bzd[i1,j1]+15])
                    sfcPrecip1L.append([sfcPrecip[i1-n1:i1+n1+1,j1-n1:j1+n1+1].mean()])
                    sfcPrecip2L.append(sfcPrecip[i1-n1:i1+n1+1,j1-n1:j1+n1+1])
                    pTypeL.append(pType[i1-n1:i1+n1+1,j1-n1:j1+n1+1])
                    
                zku1=[]
                ipass=0
                dbin1=[]
                for k in range(11):
                    zku1.append(zku[i1-5+k,j1,bzd[i1-5+k,j1]+4])
                    dbin1.append(bcf[i1-5+k,j1]-bzd[i1-5+k,j1])
                    if bzd[i1-5+k,1]>bcf[i1-5+k,j1]:
                        ipass=1
                        continue
                if ipass==1:
                    continue
                #print(np.min(dbin1))
                if np.min(dbin1)<15:
                    continue
                zKuFL2.append(zku1)
                zKuFL1.append([zku[i1-5+k,j1,bzd[i1-5+k,j1]-2]for k in range(11)])
                
                #stop
                stormTopL.append(stormTop[i1-5:i1+6,j1])
                sfcPrecipL.append(sfcPrecip[i1-5:i1+6,j1])
                #stop
                nt+=1
    print(nt)
dmL=np.array(dmL)
zcL=np.array(zcL)
sfcPrecipL=np.array(sfcPrecipL)
stormTopL=np.array(stormTopL)
stormTop2L=np.array(stormTop2L)
piaF2L=np.array(piaF2L)
piaF2L[piaF2L<0]=0
precipRateL=np.array(precipRateL)
zcL=np.array(zcL)
zKuL=np.array(zKuL)
zKuL[zKuL<0]=0
sfcPrecip1L=np.array(sfcPrecip1L)
sfcPrecip2L=np.array(sfcPrecip2L)
pTypeL=np.array(pTypeL)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xarray as xr
xarrayout=1


print("writing data out...")
if xarrayout==1:
    zData=xr.DataArray(zKuL,dims=['nt','n40','n1s'])
    sfcPrecipx=xr.DataArray(sfcPrecip1L[:,0],dims=['nt'])
    sfcPrecip2x=xr.DataArray(sfcPrecip2L[:,:],dims=['nt','n1','n1'])
    pTypex=xr.DataArray(pTypeL,dims=['nt','n1','n1'])
    stormTopx=xr.DataArray(stormTop2L,dims=['nt','n1','n1'])
    zcx=xr.DataArray(zcL,dims=['nt','n1','n1'])
    dmx=xr.DataArray(dmL,dims=['nt','n1','n1'])
    piaFx=xr.DataArray(piaF2L,dims=['nt','n1','n1','n2'])
    precipRatex=xr.DataArray(precipRateL,dims=['nt','n1','n1'])
    d={"zData":zData,"sfcPrecip":sfcPrecipx,"sfcPrecip2D":sfcPrecip2x,"pType":pTypex,\
       "zc":zcx,"dm":dmx,"stormTop":stormTopx,"precipRate":precipRatex,"piaF":piaFx}
    ds=xr.Dataset(d)
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf("zData9x9_KWAJ_t.nc", encoding=encoding)
stop
print("preparing tf model...")
scalerZKu = StandardScaler()
scalerPrec=StandardScaler()


pRateL=np.array(pRateL)
pRateL[pRateL<0]=0

#scalerZKu.fit(zKuL[:,:])
#zKu_sc=scalerZKu.transform(zKuL)[:,:]
zKu_sc=zKuL.copy()
nt,nz,nr=zKu_sc.shape
zm=[zKuL[:,:,k].mean(axis=0) for k in range(nr)]
zs=[zKuL[:,:,k].mean(axis=0) for k in range(nr)]

for k in range(nr):
    zKu_sc[:,:,k]=(zKuL[:,:,k]-zm[40])/zs[40]
    
scalerPrec.fit(sfcPrecip1L[:,:])
pRate_sc=scalerPrec.transform(sfcPrecip1L)[:,:]

from sklearn.decomposition import PCA
pca = PCA(n_components=20)
pca.fit(zKu_sc[:,:,40])
pcs=pca.transform(zKu_sc[:,:,40])
pcs_std=pcs.std(axis=0)
nm=9
npc=8
a=np.nonzero(sfcPrecip1L[:,0]>8)
X=np.zeros((nt+len(a[0]),nm,nm,npc+1),float)
it=0
for i in range(nm):
    for j in range(nm):
        pcs=pca.transform(zKu_sc[:,:,it])
        for k in range(npc):
            pcs[:,k]/=pcs_std[k]
        X[:nt,i,j,:npc]=pcs[:,:npc]
        it+=1
X[:nt,:,:,npc]=pTypeL[:,:,:]

for iadd,ipos in enumerate(len(a[0])):
    it=0
    for i in range(nm):
        for j in range(nm):
            pcs=pca.transform(zKu_sc[ipos:ipos+1,:,it])
            for k in range(npc):
                pcs[0,k]/=pcs_std[k]
        X[nt+iadd,j,i,:npc]=pcs[0,:npc]
        it+=1
    X[nt+iadd,:,:,npc]=pTypeL[ipos,:,:].T
        
ind_train, ind_test,\
    y_train, y_test = train_test_split(range(pRate_sc.shape[0]), pRate_sc,\
                                       test_size=0.5, random_state=42)

X_train=X[ind_train,:,:,:].copy()
X_test=X[ind_test,:,:,:].copy()

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop


import tensorflow as tf
model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, \
                                 kernel_regularizer=tf.keras.regularizers.L1(0.0001),input_shape=[9, 9, npc+2]))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,kernel_regularizer=tf.keras.regularizers.L1(0.0001),))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(1))
#radarProfilingCNN_3_less_reg.h5 (0.34216705905825723 -0.1722004783228311) (0.4738876406575849 0.30527402416256033)
itrain=1
if itrain==1:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  \
        loss='mse',\
        metrics=[tf.keras.metrics.MeanSquaredError()])
    history = model.fit(X_train[:,:], y_train[:,:], \
                        batch_size=32,epochs=50,
                        validation_data=(X_test[:,:], \
                                         y_test[:,:]))
    model.save("radarProfilingCNN_4_less_reg.h5")
else:
    model=tf.keras.models.load_model("radarProfilingCNN_Kwaj_less_reg.h5")

    

import numpy as np
#import sklearn

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans


nn=50
from sklearn.neighbors import KNeighborsRegressor
#neigh = KNeighborsRegressor(n_neighbors=30,weights='distance')
#neigh.fit(X_train, y_train)

#nn=50
#from sklearn.neighbors import KNeighborsRegressor
#neigh = KNeighborsRegressor(n_neighbors=50,weights='distance')
#neigh.fit(X_train, y_train)
y_=model.predict(X_test)
prec_=scalerPrec.inverse_transform(y_)[:,0]
precTruth=scalerPrec.inverse_transform(y_test)[:,0]

a=np.nonzero((precTruth-8)*(precTruth-10)<0)
bias=(prec_[a]-precTruth[a]).mean()/precTruth[a].mean()
rms=(prec_[a]-precTruth[a]).std()/precTruth[a].mean()

a2=np.nonzero((precTruth-.5)*(precTruth-1.75)<0)
bias2=(prec_[a2]-precTruth[a2]).mean()/precTruth[a2].mean()
rms2=(prec_[a2]-precTruth[a2]).std()/precTruth[a2].mean()

#print(rms,bias)

import matplotlib
plt.hist2d(prec_[:],precTruth[:],bins=np.arange(31)*0.3,norm=matplotlib.colors.LogNorm(),cmap='jet')
