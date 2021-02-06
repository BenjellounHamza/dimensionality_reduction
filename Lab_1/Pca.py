from processing_data import processing_data, plot
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.linalg import svd

data, type = processing_data()
data_centred = pd.DataFrame(StandardScaler().fit_transform(data))


"""PCA with SVD"""

# apply SVD on centred data
U, S, Vt = svd(np.transpose(data_centred), full_matrices=True)
# extract the two principales componentes
U_2 = U[:, [0, 1]]
#transform data
X_pca = np.dot(np.transpose(U_2), np.transpose(data_centred))
# to plot the figure uncomment the following line
#plot(np.transpose(X_pca), type)


# Reconstruction error of PCA:
X_reconstructed = np.dot(U_2, X_pca)
X = np.transpose(np.array(data_centred))
error_PCA = np.mean((X - X_reconstructed)**2)

"""Implemented pca in python"""

# Use implemented pca in python
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data_centred)
# to plot the figure uncomment the following line
plot(principalComponents, type)
