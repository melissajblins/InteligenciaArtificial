#https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import neighbors
from sklearn import cluster
from sklearn import metrics
import math
import matplotlib.pyplot as plt
from scipy import stats

dados = np.loadtxt('heart.csv', dtype = float, delimiter = ',', skiprows = 1)
y = dados[:, -1]
X = dados[:, 0:-1]

exp = make_pipeline(preprocessing.StandardScaler(), neighbors.KNeighborsClassifier(3))
res = cross_val_score(exp, X, y, cv = 10)
#print(np.mean(res))
#print(np.std(res))

scl = preprocessing.StandardScaler()
scl.fit(X)
X_scl = scl.transform(X)

agrup = cluster.KMeans(n_clusters = 2)
agrup.fit(X_scl)
#print(agrup.inertia_)
#print(agrup.labels_)
#print(y)
#print(metrics.silhouette_score(X_scl, agrup.labels_))

n_classes = np.unique(y).shape[0]
max_grupos = round(math.sqrt(y.shape[0]))

agrup = cluster.KMeans(n_clusters = n_classes - 1)
agrup.fit(X_scl)
eq = [agrup.inertia_]
silh = []

for i in range(n_classes, max_grupos + 1):
  agrup = cluster.KMeans(n_clusters = i)
  agrup.fit(X_scl)
  eq.append(agrup.inertia_)
  silh.append(metrics.silhouette_score(X_scl, agrup.labels_))

plt.plot(np.arange(n_classes - 1, max_grupos + 1), eq)
plt.savefig('eq')
plt.clf()

plt.plot(np.arange(n_classes, max_grupos + 1), silh)
plt.savefig('silh')
plt.clf()

n_grupos = 16
agrup = cluster.KMeans(n_clusters = n_grupos)
agrup.fit(X_scl)
grupos = agrup.labels_

agrup.predict(X_scl[0:5,:])

#print(y[grupos == 0])
#print(y[grupos == 1])
#print(y[grupos == 2])
#print(stats.mode(y[grupos == 0]))

grupos_classes = []
for i in range(0, n_grupos):
  grupos_classes.append(stats.mode(y[grupos == i])[0][0])

#Atividade:
#- Utilizar o agrupamento para classificar novos pacientes da seguinte forma:
#- 1. divida a base para treinamento e teste
#- 2. realize o agrupamento na base de treinamento
#- 3. utilize a função predict do agrupamento para classificar a base de teste (analisando a qual classe pertence o grupo indicado pelo predict)
#- 4. calcule a média de acerto
#- A divisão em treinamento e teste pode ser feita com o método holdout (como na  Prática 4), ou k-fold crossvalidation valendo bônus na nota.