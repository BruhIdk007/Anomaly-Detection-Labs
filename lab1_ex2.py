import pyod.utils.data as data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix,roc_curve
import matplotlib.pyplot as plt
X_train,X_test,Y_train,Y_test=data.generate_data(n_train=400,n_test=100,contamination=0.1)
knn=KNN(contamination=0.1)
knn.fit(X_train)
y_pred_train=knn.predict(X_train)
y_pred_test=knn.predict(X_test)
cf1=confusion_matrix(Y_train,y_pred_train)
print(cf1)
tn1=cf1[0,0]
fp1=cf1[0,1]
fn1=cf1[1,0]
tp1=cf1[1,1]
tpr1=tp1/(tp1+fn1)
tnr1=tn1/(tn1+fp1)
ba1=(tpr1+tnr1)/2
print(ba1)
fpr1,tpr1,_=roc_curve(Y_train,knn.decision_scores_)
print(fpr1)
print(tpr1)
plt.plot(fpr1,tpr1)
plt.xlabel('Fals pozitiv')
plt.ylabel('True pozitiv')
plt.title('Curba ROC la train')
plt.show()
cf2=confusion_matrix(Y_test,y_pred_test)
print(cf2)
tp2=cf2[1,1]
fn2=cf2[1,0]
fp2=cf2[0,1]
tn2=cf2[0,0]
tpr2=tp2/(tp2+fn2)
tnr2=tn2/(tn2+fp2)
ba2=(tpr2+tnr2)/2
print(ba2)
fpr2,tpr2,_=roc_curve(Y_test,knn.decision_function(X_test)) #cu decision_function obtinem scorul de decizie pentru fiecare punct.
print(fpr2)
print(tpr2)
plt.plot(fpr2,tpr2)
plt.xlabel('Fals pozitiv')
plt.ylabel('True pozitiv')
plt.title('Curba ROC la test')
plt.show()
knn1=KNN(contamination=0.3)
knn1.fit(X_train)
y_pred_train1=knn1.predict(X_train)
y_pred_test1=knn1.predict(X_test)
cf3=confusion_matrix(Y_train,y_pred_train1)
print(cf3)
tn3=cf3[0,0]
fp3=cf3[0,1]
fn3=cf3[1,0]
tp3=cf3[1,1]
tpr3=tp3/(tp3+fn3)
tnr3=tn3/(tn3+fp3)
ba3=(tpr3+tnr3)/2
print(ba3)
fpr3,tpr3,_=roc_curve(Y_train,knn1.decision_scores_)# cu decision_scores obtinem scorul de decizie pentru fiecare punct.
print(fpr3)
print(tpr3)
plt.plot(fpr3,tpr3)
plt.xlabel('Fals pozitiv')
plt.ylabel('True pozitiv')
plt.title('Curba ROC la train')
plt.show()
cf4=confusion_matrix(Y_test,y_pred_test1)
print(cf4)
tp4=cf4[1,1]
fn4=cf4[1,0]
fp4=cf4[0,1]
tn4=cf4[0,0]
tpr4=tp4/(tp4+fn4)
tnr4=tn4/(tn4+fp4)
ba4=(tpr4+tnr4)/2
print(ba4)
fpr4,tpr4,_=roc_curve(Y_test,knn1.decision_function(X_test)) #cu decision_function obtinem scorul de decizie calcuat in fiecare punct de train utilizand functia de decizie a modelului KNN.
print(fpr4)
print(tpr4)
plt.plot(fpr4,tpr4)
plt.xlabel('Fals pozitiv')
plt.ylabel('True pozitiv')
plt.title('Curba ROC la test')
plt.show()
#daca schimbam rata de contaminare putem observa faptul ca numarul de fn creste in detrimentul fp, care scade.

