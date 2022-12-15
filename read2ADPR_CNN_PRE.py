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
    return sfcPrecip,hzero,pType,stormTop,bzd,zku,bcf,precipRate,dm,zkuc,fh
eRad=6357e3

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

fh=Dataset("zData9x9_KWAJ_t.nc")
zKuL=fh["zData"][:]
stormTop=fh["stormTop"][:]
pTypeL=fh["pType"][:]
sfcPrecip1L=fh["sfcPrecip"][:]
sfcPrecip1L=sfcPrecip1L[:,np.newaxis]
print("preparing tf model...")
scalerZKu = StandardScaler()
scalerPrec=StandardScaler()
piaF=fh["piaF"][:,:,:,0]

#539424
zKu9x9=np.swapaxes(zKuL,1,2)
nt,nz,nr=zKuL.shape
nm=9
zKu9x9=zKu9x9.reshape(nt,nm,nm,nz)
zmax=[zKu9x9[k,:,:,0:25].max(axis=-1) for k in range(nt)]
zmax=np.array(zmax)
#scalerZKu.fit(zKuL[:,:])
#zKu_sc=scalerZKu.transform(zKuL)[:,:]
zKu_sc=zKuL.copy()
nt,nz,nr=zKu_sc.shape
zm=[zKuL[:,:,k].mean(axis=0) for k in range(nr)]
zs=[zKuL[:,:,k].mean(axis=0) for k in range(nr)]

piaF+=np.random.randn(nt,nm,nm)*3
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
X=np.zeros((nt+len(a[0]),nm,nm,npc+4),float)
it=0
for i in range(nm):
    for j in range(nm):
        pcs=pca.transform(zKu_sc[:,:,it])
        for k in range(npc):
            pcs[:,k]/=pcs_std[k]
        X[:nt,i,j,:npc]=pcs[:,:npc]
        it+=1
X[:nt,:,:,npc]=pTypeL[:,:,:]/3.0
X[:nt,:,:,npc+1]=(stormTop-5.2e3)/4.2e3
X[:nt,:,:,npc+2]=(zmax-19.8)/8.23
X[:nt,:,:,npc+3]=(piaF[:nt,:,:]-0.4)/2.
for iadd,ipos in enumerate(a[0]):
    it=0
    for i in range(nm):
        for j in range(nm):
            pcs=pca.transform(zKu_sc[ipos:ipos+1,:,it])
            for k in range(npc):
                pcs[0,k]/=pcs_std[k]
        X[nt+iadd,j,i,:npc]=pcs[0,:npc]
        it+=1
    X[nt+iadd,:,:,npc]=(pTypeL[ipos,:,:]/3.0).T
    X[nt+iadd,:,:,npc+1]=(stormTop[ipos,:,:].T-5.2e3)/4.2e3    

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
                                 kernel_regularizer=tf.keras.regularizers.L1(0.0001),input_shape=[9, 9, npc+4]))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,kernel_regularizer=tf.keras.regularizers.L1(0.0001),))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(1))
#radarProfilingCNN_3_less_reg.h5 (0.34216705905825723 -0.1722004783228311) (0.4738876406575849 0.30527402416256033)
#radarProfilingCNN_3_less_reg.h5 tf.keras.regularizers.L1(0.0001
itrain=1
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
    model.save("radarProfilingCNN_KWAJ1_PIA3dB.h5")
else:
    model=tf.keras.models.load_model("radarProfilingCNN_KWAJ_PIA3dB.h5")

    

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
ax=plt.subplot(111)
c=plt.hist2d(precTruth[:],prec_[:],bins=np.arange(34)*0.3,norm=matplotlib.colors.LogNorm(),cmap='jet')
ax.set_aspect('equal')
plt.xlabel('Truth (mm/h)')
plt.ylabel('Estimated (mm/h)')
plt.title('50 km average precipitation rate\nKuPR')
cbar=plt.colorbar(c[-1])
cbar.ax.set_title('Counts')
plt.savefig('50kmPrecipDistrib.png')


plt.figure()


labels = ['1.0 mm/h', '10.0 mm/h']
rmsL = (np.array([rms2,rms])*100.).astype(int)
biasL = (np.array([bias2,bias])*100.).astype(int)

x = np.arange(len(labels))  # the label locations
width = 0.39  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x -0.01- width/2, rmsL, width, label='RMS')
rects2 = ax.bar(x +0.01+ width/2, biasL, width, label='Bias')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Normalized scores [%]')
ax.set_title('')
ax.set_xticks(x, labels)
ax = plt.gca()
ax.spines["bottom"].set_position(("data", 0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
plt.title('KuPR performance')
plt.savefig('KuPR.png')

fig.tight_layout()



