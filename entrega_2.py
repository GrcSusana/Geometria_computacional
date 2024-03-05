# Entrega 2 - GCOM - Susana Garcia Martin

import numpy as np
import os

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from scipy.spatial import Voronoi, voronoi_plot_2d

import matplotlib.pyplot as plt


# directorio en el que estoy trabajando
os.getcwd()

# importo los archivos 
archivo1 = "Personas_de_villa_laminera.txt" 
archivo2 = "Franjas_de_edad.txt"
X = np.loadtxt(archivo1,skiprows=1)
Y = np.loadtxt(archivo2,skiprows=1)
labels_true = Y[:,0]


# Apartado 1

# coeficiente de Silhouette para diferente numero de vecindades k = {2,3,...,15}
# usando el algoritmo de K-Means


def coefs_kmeans(k1,k2):
    l = []
    for i in range(k1,k2+1):
        kmeans = KMeans(n_clusters=i, random_state=0, n_init='auto').fit(X)
        labels_kmeans = kmeans.labels_
        silhouette = metrics.silhouette_score(X, labels_kmeans)
        l.append(silhouette)
    return l


l_coefs_kmeans = coefs_kmeans(2,15)


# Mostrar una grafica del valor del c. Silhouette en funcion de k y decidir con
# ello cual es el numero optimo de vecindades


plt.plot(range(2,16), l_coefs_kmeans)
plt.xlabel("Número de vecindades")
plt.ylabel("Coeficiente de Silhouette")
plt.show()


# En una segunda grafica muestra la clasificacion (clusters) resultante con
# diferentes colores y representa el diagrama de Voronoi en esa misma grafica

n_clusters_kmeans=3

kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=0, n_init='auto').fit(X)
labels = kmeans.labels_
silhouette_kmeans = metrics.silhouette_score(X, labels)

print("Coeficiente de Silhouette para k-means con tres vecindades: %0.3f" % silhouette_kmeans)


problem = np.array([[1/2,0],[0,-3]])
clases_pred = kmeans.predict(problem)
print(clases_pred)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]


plt.figure(figsize=(8,4))

# centroides = kmeans.cluster_centers_
# voronoi = Voronoi(centroides)
# voronoi_plot_2d(vor = voronoi)

plt.xlim(-3,4)
plt.ylim(-4,4)

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=5)
    

# plt.plot(problem[:,0],problem[:,1],'o', markersize=11, markeredgecolor='k', markerfacecolor="red")

plt.title('Número de vecindades para k-means: %d' % n_clusters_kmeans)
plt.show()




# Apartado 2 

# Obten el c. Silhouette para el mismo sistema A usando ahora el algoritmo
# DBSCAN con la metrica euclidean y luego con la manhattan
# En este caso el parametro que debemos explorar es el umbral de distancia
# epsilon perteneciente a (0.1, 0.4) fijando el numero de elementos minimo en
# n0 = 10.
# Comparar graficamente con el resultado del apartado anterior.

# Coeficiente de Silhouette usando el algoritmo de DBSCAN con la metrica euclidea

n0 = 10

l_umbral = np.arange(0.1, 0.4, 0.05)

def coefs_s_dbscan(metrica):
    l = []
    for epsilon in l_umbral:
        db = DBSCAN(eps=epsilon, min_samples=n0, metric=metrica).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels_dbscan = db.labels_
        silhouette = metrics.silhouette_score(X, labels_dbscan)
        l.append(silhouette)
    return l

#Hago una grafica comparando los valores de los coeficientes de silhouette para
# varios valores del umbral de distancia y para ambas metricas

l_coefs_dbscan_euclidean = coefs_s_dbscan('euclidean')
plt.plot(l_umbral, l_coefs_dbscan_euclidean, label='Métrica euclídea')
l_coefs_dbscan_manhattan = coefs_s_dbscan('manhattan')
plt.plot(l_umbral, l_coefs_dbscan_manhattan, label='Métrica Manhattan')
plt.legend(loc="lower right")
plt.xlabel("Umbral de distancia")
plt.ylabel("Coeficiente de Silhouette")
plt.show()


# viendo el grafico, con el coef de silhouette decido el valor optimo de
# epsilon para cada una de las metricas

epsilon_euclidean = 0.3

epsilon_manhattan = 0.4


# A continuacion hallo el numero optimo de clusters para la metrica euclidea
# y hago una grafica

db = DBSCAN(eps=epsilon_euclidean, min_samples=n0, metric='euclidean').fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_euclidean = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_euclidean = list(labels).count(-1)

print('********** MÉTRICA EUCLÍDEA **********')
print('Número estimado de vecindades para la métrica euclídea: %d' % n_clusters_euclidean)
print('Número estimado de puntos de ruido: %d' % n_noise_euclidean)
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Coeficiente de Silhouette: %0.3f"
      % metrics.silhouette_score(X, labels))

silhouette_euclidean = metrics.silhouette_score(X, labels)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(8,4))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)

plt.title('Número estimado de vecindades para DBSCAN con métrica euclídea: %d' % n_clusters_euclidean)
plt.show()


# Ahora hago lo mismo pero para la métrica manhattan

db = DBSCAN(eps=epsilon_manhattan, min_samples=n0, metric='manhattan').fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_manhattan = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_manhattan = list(labels).count(-1)

print('********** MÉTRICA MANHATTAN **********')
print('Número estimado de vecindades para la métrica Manhattan: %d' % n_clusters_manhattan)
print('Número estimado de puntos de ruido: %d' % n_noise_manhattan)
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Coeficiente de Silhouettet: %0.3f"
      % metrics.silhouette_score(X, labels))

silhouette_manhattan = metrics.silhouette_score(X, labels)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(8,4))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)

plt.title('Número estimado de vecindades para DBSCAN con métrica Manhattan: %d' % n_clusters_manhattan)
plt.show()


# Apartado 3

# ¿De que franja de edad diriamos que son las personas con coordenadas 
# a = (1/2, 0) y b = (0,-3)? Comprueba tu respuesta con la funcion 
# kmeans.predict


problem = np.array([[1/2,0],[0,-3]])
clases_pred = kmeans.predict(problem)
print(clases_pred)

# a pertenece a la categoria 2
# b pertenece a la categoria 0
