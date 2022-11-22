from netCDF4 import Dataset
import glob
import numpy as np
fs=glob.glob("collocL/*nc")
nt=0
piaKuL=[]
tbL=[]
kern=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
nt2=0

for f in fs:
    print(f)
    fh=Dataset(f)
    reliab=fh["reliabKuF"][:]
    bzd=fh["bzd"][:]
    bcf=fh["bcf"][:]
    a1=np.nonzero((bzd-150)*(bzd-100)<0)
    a2=np.nonzero(reliab[a1]==1)
    piaKu=fh["piaKu"][:]
    zKu=fh["zKu"][:]
    zKuL=[]
    zKu[zKu<0]=0
    X=[]
    y=[]
    for i in a1[0]:
        if bcf[i]<bzd[i]+20:
            continue
        if not(reliab[i]==1 or reliab[i]==1):
            continue
        x1=[]
        x1.extend(zKu[i,bzd[i]-40:bzd[i]+20:2])
        X.append(x1)
        y.append(piaKu[i])
    tbL.extend(X)
    piaKuL.extend(y)
   
#stop
from sklearn import neighbors
k=50
tbL=np.array(tbL)
piaKuL=np.array(piaKuL)
a=np.nonzero(piaKuL==piaKuL)
piaKuL=piaKuL[a]
tbL=tbL[a]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(\
    np.array(tbL)[:,:], np.array(piaKuL)[:,np.newaxis], test_size=0.33, random_state=42)

n_neighbors = k



from tensorflow import keras
from keras.layers import Dense,BatchNormalization,Dropout
nin=X_train.shape[1]
model = keras.models.Sequential()
model.add(Dense(8,input_shape=(nin,),activation="relu"))
model.add(BatchNormalization())
model.add(Dense(8,activation="relu"))
#model.add(Dropout(0.25))
model.add(Dense(8,activation="relu"))
model.add(BatchNormalization())
model.add(Dense(8,activation="relu"))
model.add(BatchNormalization())
model.add(Dense(2,activation="linear"))
model.compile(loss='mae', optimizer='adam', metrics=['mae'])

from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
scalery = StandardScaler()

X_trainNN=scalerX.fit_transform(X_train[:,:])
y_trainNN=scalery.fit_transform(y_train[:,:])
X_testNN=scalerX.transform(X_test[:,:])
y_testNN=scalery.transform(y_test[:,:])

print(scalerX.mean_)
print(scalerX.scale_)
print(scalery.mean_)
print(scalery.scale_)

import pickle
pickle.dump({"scalerX":scalerX,"scalery":scalery},open("scalers_32_Land.pklz","wb"))
model.fit(X_trainNN, y_trainNN, epochs=40, batch_size=64)
model.save('piaTbZ_with_SF_SRT_Land.h5')

import pickle

y_=model.predict(X_testNN)
pia_=scalery.inverse_transform(y_)

import matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
ax=plt.subplot(122)
piaHist2d=plt.hist2d(y_test[:,0],pia_[:,0],bins=np.arange(40)*0.25,norm=matplotlib.colors.LogNorm(),cmap='jet')
ax.set_aspect('equal')
plt.xlabel('Reference PIA(Ku)[dB]')
plt.ylabel('ML PIA(Ku)[dB]')
plt.xlim(0,7)
plt.ylim(0,7)
cbar=plt.colorbar(piaHist2d[-1],orientation='horizontal')
cbar.ax.set_title('Counts')

plt.tight_layout()
plt.savefig('ML_PIA_32_Land.png')
