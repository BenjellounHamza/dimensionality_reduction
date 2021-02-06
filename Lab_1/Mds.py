from processing_data import processing_data, plot
from numpy import linalg as lg
from scipy.spatial import distance_matrix
import numpy as np


"""
Define features importance:
"""

def compute_weight(X, y):
  # define the first model
  model = KNeighborsRegressor()
  # fit the model
  model.fit(X, y)
  # perform permutation importance
  results = permutation_importance(model, X, y, scoring='neg_mean_squared_error')
  # get the first importance
  importance_1 = results.importances_mean

  # define the second model
  model = RandomForestRegressor()
  # fit the model
  model.fit(X, y)
  # get the second importance
  importance_2 = model.feature_importances_

  #combine the two importances
  importance = []
  for i in range(len(importance_1)):
    importance.append((abs(importance_1[i]) + abs(importance_2[i]))/2)
  # summarize feature importance
  for i, v in enumerate(importance):
	  print('Feature: %0d, Score: %.5f' % (i,v))
  # plot feature importance
  pyplot.bar([x for x in range(len(importance))], importance)
  pyplot.show()
  return importance

def MDS(df, k, predef_distance = False, weighted = False):
  if predef_distance:
    # if we give distance to MDS
    D = df
  else:
    #compute distances
    if weighted:
      #if we want to add importance of features to data
      weight_data = df.copy()
      importance = compute_weight(weight_data, type)
      for i, imp in enumerate(importance):
        weight_data[weight_data.columns[i]] = weight_data[weight_data.columns[i]].apply(lambda x: x * (imp + 1))
      D = distance_matrix(weight_data.values, weight_data.values)
    else:
      D = distance_matrix(df.values, df.values)
  n = len(D)
  #compute S by double centring
  ones = np.ones((n, n))
  S = - (1/2) * (D - (1/n) * np.dot(D, ones) - (1/n) * np.dot(ones, D) + (1/n**2) * np.dot(ones, np.dot(D, ones)))
  #EigenDecompostion
  eigenvalues, eigenvectors = lg.eig(S)
  eigenvalues, eigenvectors = np.abs(np.real(eigenvalues)), np.real(eigenvectors)
  lambdas = np.sqrt(np.diag(eigenvalues))
  X_mds = np.dot(np.eye(k, n), np.dot(lambdas, np.transpose(eigenvectors)))
  return np.transpose(X_mds)

data, type = processing_data()
X_mds = MDS(data, 2, False, False)
# to plot the figure uncomment the following line
#plot(X_mds, type)
