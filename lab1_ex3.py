import pyod.utils.data as data
import numpy as np
X_train,X_test,Y_train,Y_test=data.generate_data(n_train=1000,n_test=0,contamination=0.1,n_features=1) #avem scorul de contaminare 0.1, asta inseamna ca 10% din date sunt outlieri
#pentru z-score folosim |x-medie|/std
#lucram direct cu x_train
med=np.mean(X_train)
std=np.std(X_train)
z_scores=np.abs(X_train-med)/std
z_scores=z_scores.flatten() #flatten pentru a transforma matricea intr-un vector 1-dimensional, fara asta are acelasi shape precum X_train
#print(z_scores)
prag=np.quantile(z_scores,0.9) #90% din date sunt inliers, ne intereseaza outliers
#for i in range(len(z_scores)):
    #if z_scores[i]>=prag:
        #print(f'Punctul {X_train[i]} este outlier')
    #else:
        #print(f'Punctul {X_train[i]} este inlier')
# acum, avem nevoie de a gasi etichetele reale pentru a putea calcula balanced accuracy, pe acestea le gasim in Y_train
tp=0
fn=0
fp=0
tn=0
for i in range(len(Y_train)):
    if Y_train[i]==1:
        if z_scores[i]>=prag:
            tp+=1 #outlier identificat corect
        else:
            fn+=1 #outlier identificat drept inlier
    else:
        if z_scores[i]>=prag:
            fp+=1 #inlier identificat drept outlier
        else:
            tn+=1 #inlier identificat corect
tpr=tp/(tp+fn)
tnr=tn/(tn+fp)
ba=(tpr+tnr)/2
print(f'Balanced accuracy:{ba}')