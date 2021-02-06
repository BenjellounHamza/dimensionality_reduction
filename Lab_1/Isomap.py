from processing_data import processing_data, plot
import pandas as pd
from Graph import Graph
import numpy as np
from Mds import MDS
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.graph_shortest_path import graph_shortest_path


"""**Isomap**"""

def isomap(df, p, k):
  X = df.to_numpy()
  graph = kneighbors_graph(X, p, mode='distance')
  A = kneighbors_graph(X, p, mode='connectivity').toarray()
  distances = graph_shortest_path(graph, directed = False, method = 'FW')
  X = MDS(distances, k, True, False)
  cc = Graph(A).connected_components()
  if(len(cc) != 1):
    print("The graph is disconnected. Therefore we will have", len(cc), "separated graphs")
  return X

data, type = processing_data()
X_iso = isomap(data, 16, 2)
# to plot the figure uncomment the following line
#plot(X_iso, type)
