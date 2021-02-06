from Isomap import isomap
from Mds import compute_weight, MDS
from processing_data import processing_data, plot
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

"""**Compute error**
As we demonstrate in the exercice number 5, PCA and classical MDS are equivalent. We will compare just PCA and Isomap. The problem is that PCA try to keep the euclidean distance between points constant. However Isomap try to keep the geodesic distance between points constant. So let first compare the two models by computing the geodisc error.
"""

data, type = processing_data()
data_centred = pd.DataFrame(StandardScaler().fit_transform(data))

#compute errors (compare isomap to pca)
number_of_neighbors = 16

X_iso = isomap(data, number_of_neighbors, 2)
pca = PCA(n_components=2)
X_pca =  pca.fit_transform(data_centred)
X_mds = MDS(data, 2)

nbrs = NearestNeighbors(n_neighbors=number_of_neighbors, algorithm='brute').fit(data)
distances, indices = nbrs.kneighbors(data)

nbrs = NearestNeighbors(n_neighbors=number_of_neighbors, algorithm='brute').fit(data_centred)
distances, indices_centred = nbrs.kneighbors(data_centred)

nbrs = NearestNeighbors(n_neighbors=number_of_neighbors, algorithm='brute').fit(X_iso)
distances, indices_iso = nbrs.kneighbors(X_iso)

nbrs = NearestNeighbors(n_neighbors=number_of_neighbors, algorithm='brute').fit(X_pca)
distances, indices_pca = nbrs.kneighbors(X_pca)

nbrs = NearestNeighbors(n_neighbors=number_of_neighbors, algorithm='brute').fit(X_mds)
distances, indices_mds = nbrs.kneighbors(X_mds)

error_iso = 0
error_pca = 0
error_mds = 0

for i in range(len(indices)):
  error_iso += len(set(indices[i]) - set(indices_iso[i]))
  error_pca += len(set(indices_centred[i]) - set(indices_pca[i]))
  error_mds += len(set(indices[i]) - set(indices_mds[i]))

print("Local structure  error:")
print("isomap error = ", error_iso/(len(indices)*number_of_neighbors))
print("mds error = ", error_mds/(len(indices)*number_of_neighbors))
print("pca error = ", error_pca/(len(indices)*number_of_neighbors))
