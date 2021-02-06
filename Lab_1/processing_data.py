import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from matplotlib import pyplot

"""Processing data"""
def processing_data():
    data = pd.read_csv("zoo.data", header=None)
    names = data.pop(0)
    type = data.pop(17).to_numpy()
    data = pd.get_dummies(data, columns = [13])
    return data, type

def plot(X, type):
  df = pd.DataFrame(data = X,
                  columns = ['first ax', 'second ax'])
  # plot different types with differents color.
  mapping= {1: "type_1", 2: "type_2", 3:"type_3", 4:"type_4", 5:"type_5", 6:"type_6", 7:"type_7", 8: "brown"}

  fig = px.scatter(df, x='first ax', y='second ax', color= [mapping[t] for t in type])
  fig.show()

def compute_duplicate(data):
    # counting the duplicates
    columns = [col for col in data.columns]
    duplicated = data.pivot_table(index = columns, aggfunc ='size')
    return duplicated
