from netCDF4 import Dataset
import glob
import numpy as np
fs=glob.glob("collocL/*nc")
nt=0
piaKuL=[]
tbL=[]
kern=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
nt2=0
X=[]
zsfcL=[]
zsfcmL=[]
piaKuDL=[]

zKuDBaseL=[]
zKaDBaseL=[]
pTypeDBaseL=[]
reliabDBaseL=[]
piaKuDBaseL=[]
bzdDBaseL=[]
bcfDBaseLL=[]
jDBaseL=[]

for f in fs:
    print(f)
    fh=Dataset(f)
    reliab=fh["reliabKuF"][:]
    reliabd=fh["reliabF"][:]
    bzd=fh["bzd"][:]
    bcf=fh["bcf"][:]
    a1=np.nonzero((bzd-150)*(bzd-100)<0)
    a2=np.nonzero(reliab[a1]==1)
    piaKu=fh["piaKu"][:]
    piaKu_d=fh["pia"][:,0]
    zKu=fh["zKu"][:]
    zKa=fh["zKa"][:]
    pType=fh["pType"][:]
    zKuL=[]
    zKu[zKu<0]=0
    indj=fh["jL"][:]
    #stop
    for i in a1[0]:
        if bcf[i]<bzd[i]+20:
            continue
        if pType[i]!=2:
            continue
        if reliab[i]==3 or reliab[i]==3:
            continue
        if piaKu[i]!=piaKu[i]:
            continue
        x1=[]
        x1.extend(zKu[i,bzd[i]-40:bzd[i]+20:2])
        piaKuL.append(piaKu[i])
        if reliabd[i]==1 or reliabd[i]==2:
            piaKuDL.append(piaKu_d[i])
        else:
            piaKuDL.append(np.log(-1))
        #if piaKuL[-1]!=piaKuL[-1]:
        #    stop
        zsfcL.append(zKu[i,bcf[i]]+piaKu[i])
        zsfcmL.append(zKu[i,bcf[i]])
        X.append(x1)
    
   
   
#stop
from sklearn import neighbors
k=50
nc=10
from sklearn.model_selection import train_test_split
X=np.array(X)
piaKuL=np.array(piaKuL)
piaKuDL=np.array(piaKuDL)
a=np.nonzero(piaKuL==piaKuL)
zsfcL=np.array(zsfcL)
zsfcmL=np.array(zsfcmL)
piaKuL=piaKuL[a]
zsfcL=zsfcL[a]
zsfcmL=zsfcmL[a]
piaKuDL=piaKuDL[a]
X=X[a]
ind=range(X.shape[0])
X_train, X_test, ind_train, ind_test = train_test_split(X, ind, \
                                                    test_size=0.33, random_state=42)
y_test=zsfcL[ind_test]
y_train=zsfcL[ind_train]
pia_test=piaKuL[ind_test]
pia_train=piaKuL[ind_train]
piaD_test=piaKuDL[ind_test]
piaD_train=piaKuDL[ind_train]
zsfc_test=zsfcmL[ind_test]
zsfc_train=zsfcmL[ind_train]

nc=30
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=nc, weights='uniform')
neigh.fit(X_train, y_train)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import numpy as np

kmeans = KMeans(n_clusters=nc, random_state=0).fit(X_train)
zKuCL=[]
piaKuCL=[]
piaKuDCL=[]
piaKuCstdL=[]
piaKuL=np.array(piaKuL)
ntL=[]
zsfcCL=[]
zsfcL=np.array(zsfcL)
sig=4
kgainCL=[]

for i in range(nc):
    a=np.nonzero(kmeans.labels_==i)
    zKuCL.append(X_train[a].mean(axis=0))
    b=np.nonzero(pia_train[a]==pia_train[a])
    piaKuCL.append(pia_train[a][b].mean(axis=0))
    c=np.array(piaD_train[a]==piaD_train[a])
    piaKuDCL.append(piaD_train[a][c].mean(axis=0))
    piaKuCstdL.append(pia_train[a][b].std(axis=0))
    ntL.append(len(b[0]))
    b=np.nonzero((zsfc_train[a]-10)*(zsfc_train[a]-60)<0)
    covXYa=np.cov(X_train[a].T,pia_train[a].T)
    covXX=covXYa[0:-1,0:-1]
    covXY=covXYa[-1,0:-1]
    covXX+=np.eye(30)*sig
    invCovXX=np.linalg.inv(covXX)
    kgain=np.dot(covXY,invCovXX)
    #print(kgain)
    kgainCL.append(kgain)
    zsfcCL.append(zsfc_train[a][b].mean())

deepC=np.argsort(piaKuCL)[:]

deepCs=[]
for i in deepC[:]:
    if piaKuCL[i]>3.4:
        plt.plot(zKuCL[i][::-1],range(X.shape[1]))
        print(piaKuCL[i],piaKuDCL[i],piaKuCstdL[i],ntL[i],zsfcCL[i])
        deepCs.append(i)

profClass=kmeans.predict(X_test)

import pickle
pickle.dump({"kmeans":kmeans},open("kmeansZKu.pklz","wb"))

y_=[]
y12=[]
for k,ip in enumerate(profClass):
    dpia=np.dot(kgainCL[ip],zKuCL[ip]-X_test[k,:])
    pia_=piaKuCL[ip]-0.975*dpia
    if ip not in deepCs:
        y12.append([pia_,pia_test[k]])
    y_.append(pia_)

stop
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()
scalery = StandardScaler()

X_trainNN=scalerX.fit_transform(X_train[:,:])
y_trainNN=scalery.fit_transform(pia_train[:,np.newaxis])
X_testNN=scalerX.transform(X_test[:,:])
y_testNN=scalery.transform(pia_test[:,np.newaxis])

import pickle
pickle.dump({"scalerX":scalerX,"scalery":scalery},open("scalers_30_Land.pklz","wb"))
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

model.fit(X_trainNN, y_trainNN, epochs=40, batch_size=64)
model.save('pia_SRT_Land.h5')
piann_=scalery.inverse_transform(ynn_)

import matplotlib.pyplot as plt
import matplotlib
plt.figure()
plt.hist2d(y_[:],pia_test[:],bins=0+np.arange(30)*0.5,norm=matplotlib.colors.LogNorm(),cmap='jet')

plt.figure()

plt.hist2d(piann_[:],pia_test[:],bins=0+np.arange(30)*0.5,norm=matplotlib.colors.LogNorm(),cmap='jet')
