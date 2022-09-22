import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

url = "C:/Users/liork/Downloads/machine.data"
names = [
    "Vendor",
    "Model",
    "MYCT",
    "MMIN",
    "MMAX",
    "CACH",
    "CHMIN",
    "CHMAX",
    "PRP",
    "ERP",
    "Target",
]

data = pd.read_csv(url, names = names)



X = data.drop(columns=['Vendor','Target','Model', 'ERP'])
Y = data['Target']

metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
print('список метрик:', end = ' ')
print(str(metrics) + '\nметрика:', end = ' ')
metricIndex = int(input()) - 1
print('значение K:', end = ' ')
k = int(input())

kNeighbors = KNeighborsClassifier(n_neighbors = k, metric = metrics[metricIndex])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
kNeighbors.fit(X_train,Y_train)
Y_pred = kNeighbors.predict(X_test)

score = cross_val_score(kNeighbors, X_test, Y_pred)
print('метрика {}, '.format(metrics[metricIndex]), end = ' ')
print('коэффициент K - {}, '.format(k), end = ' ')
print('аккуратность алгоритма {}%'.format(int(round(np.mean(score), 2) * 100)))

