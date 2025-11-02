import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as pyplot
from numpy import linspace
from tqdm import tqdm
import math
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Figuring out how to import uci datasets for use with pytorch

# fetch dataset 
adult = fetch_ucirepo(id=2)
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 

# Testing data is a subset of all data. The ucimlrepo library doesn't
# differentiate between train and test, although the downloaded .data files do.
# Luckily it seems to be the first ~32k is set for training and last ~16k set
# for testing based on looking at the .data files.
  

# X.columns is a list of the variables collected
# object type = Index
print("\n\nX.columns and type")
print(X.columns)
print(type(X.columns))


# The axes of the X dataframe. List object.
# First is a RangeIndex object (labels of 0-44842 of the individuals in the 
# dataset. These labels are used in X.loc[label] to grab an individual.)
# Second is the Index object grabbed by X.columns. Labels of variables.
print("\n\nX.axes and type")
print(X.axes)
print(type(X.axes))

# metadata 
#print("\n\nMETADATA")
#print(adult.metadata) 
  
#variable information. Gives variable names and possible values in dataset
print("\n\nVARIABLES")
print(adult.variables)

# All people
print("\n\nINDIVIDUALS")
print(X)

# Changing individual 0:
# X.loc[0] is how we grab an individual, it seems. pandas Series object.
# Series are kinda immutable one-dimensional nd arrays with axis labels-
# must create a deep copy, setting things equal creates a view on original object
print("\n\nold X.loc[0]")
print(X.loc[0])
newX = X.copy(deep = True)

# Update: age to 28, education to 11th, race to Asian-Pac-Islander
# Series are dict-like (noted in documentation)
newX['age'] = 28
newX['education'] = '11th'
newX['race'] = 'Asian-Pac-Islander'

# Print updated individual 0
print("\n\nnew X.loc[0]")
print(newX.loc[0])

# Reprint old X.loc[0] to display no change to original data
print("\n\nReprinting old X.loc[0]")
print(X.loc[0])