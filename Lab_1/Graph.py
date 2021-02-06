from collections import defaultdict

class Graph:

  def __init__(self, adjmatrix):
    #graph can be defined by adjacency Matrix
    self.adjmatrix = adjmatrix
    #dictionary map each point to their neighbors.
    self.adjdict = {}
    for i, point in enumerate(self.adjmatrix):
      self.adjdict[i] = [j for j in range(len(point)) if point[j] != 0]


  def connected_components(self):
    # compute connected componeents of the graph
    visited = defaultdict(lambda: 0, {})
    cc = []
    for i in range(len(self.adjmatrix)):
      if visited[i] == 0:
        temp = []
        cc.append(self.DFS(temp, i, visited))
    return cc

  def DFS(self, temp, i, visited):
    visited[i] = 1
    temp.append(i)
    for j in self.adjdict[i]:
      if visited[j] == 0:
        # j is linked to 'point' and is not visited
        self.DFS(temp, j, visited)
    return temp
