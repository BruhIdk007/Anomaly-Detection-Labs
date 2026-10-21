import pyod.utils.data as data
import matplotlib.pyplot as plt
X_train,_,Y_train,_=data.generate_data(n_train=400,n_test=100,contamination=0.1)
for i in range(len(Y_train)):
    if Y_train[i]==1: #vector 1-dimensional, prin 1 verificam daca punctul este outlier
        plt.scatter(X_train[i,0],X_train[i,1],color='red',zorder=2,s=50) #outliers
    else:
        plt.scatter(X_train[i,0],X_train[i,1],color='blue',zorder=1,s=50) #inliners
plt.show()
