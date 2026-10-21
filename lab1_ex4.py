import pyod.utils.data as data
import numpy as np
from numpy import linalg as la
med=np.array([2,4])
cov=np.array([[2,0.5],[0.5,1]])
X=np.random.multivariate_normal([0,0],np.eye(2),1000)#generam 1000 de puncte din distributia normala multivariata cu media si covarianta calculate anterior
X[900:]+=np.array([0.1,0.1]) #adaugam outlieri, cu cat deplasam mai putin cu atat accuracy scade
Y_train=np.array([0]*900+[1]*100) #etichetele reale pentru punctele generate anterior, 0 pentru inliers si 1 pentru outliers
ordine=np.random.permutation(len(X)) #generam o permutare aleatoare a indicilor pentru a amesteca datele
X=X[ordine]
Y_train=Y_train[ordine]
L=la.cholesky(cov) #folosim factorizarea Cholesky pentru a obtine matricea L,astfel incat cov=L*L.T
y=np.dot(X,L.T)+med #transformam punctele generate anterior pentru a obtine puncte care respecta distributia normala multivariata 
z_scores=[]
for i in range(len(y)):
    z_scores.append(np.sqrt((y[i]-med).T@la.inv(cov)@(y[i]-med)))#calculam scorul z pentru fiecare punct folosind formula z=(x-medie)/std, unde std este calculat folosind matricea de covarianta
z_scores=np.array(z_scores) #transformam z-scores in vector 1D
prag=np.quantile(z_scores,0.9) #90% din date sunt inliers
tp=0
fn=0
tn=0
fp=0
for i in range(len(z_scores)):
    if Y_train[i]==1:
        if z_scores[i]>=prag:
            tp+=1
        else:
            fn+=1
    else:
        if z_scores[i]>=prag:
            fp+=1
        else:
            tn+=1
tpr=tp/(tp+fn)
tnr=tn/(tn+fp)
ba=(tpr+tnr)/2
print(f'Balanced accuracy:{ba}')
