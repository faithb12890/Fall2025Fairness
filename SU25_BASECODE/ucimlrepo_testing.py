import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as pyplot
from numpy import linspace
from tqdm import tqdm
import math
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Figuring out how to import uci datasets for use with pytorch

# fetch dataset 
AdultPandas = fetch_ucirepo(id=2)
  
# data (as pandas dataframes) 
pandasX = AdultPandas.data.features 
pandasy = AdultPandas.data.targets

print("\n\nX and y types")
print(f"type X: {type(pandasX)}")
print(f"type y: {type(pandasy)}")

# Testing data is a subset of all data. The ucimlrepo library doesn't
# differentiate between train and test, although the downloaded .data files do.
# Luckily it seems to be the first ~32k is set for training and last ~16k set
# for testing based on looking at the .data files. 
  

# X.columns is a list of the variables collected
# object type = Index
print("\n\nX.columns and type")
print(pandasX.columns)
print(type(pandasX.columns))


# The axes of the X dataframe. List object.
# First is a RangeIndex object (labels of 0-44842 of the individuals in the 
# dataset. These labels are used in X.loc[label] to grab an individual.)
# Second is the Index object grabbed by X.columns. Labels of variables.
print("\n\nX.axes and type")
print(pandasX.axes)
print(type(pandasX.axes))

# metadata 
#print("\n\nMETADATA")
#print(adult.metadata) 
  
#variable information. Gives variable names and possible values in dataset
print("\n\nVARIABLES")
print(AdultPandas.variables)

# All people
print("\n\nINDIVIDUALS")
print(pandasX)

# Changing individual 0:
# X.loc[0] is how we grab an individual, it seems. pandas Series object.
# Series are kinda immutable one-dimensional nd arrays with axis labels-
# must create a deep copy, setting things equal creates a view on original object
print("\n\nold X.loc[0]")
print(pandasX.loc[0])
newX = pandasX.copy(deep = True)

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
print(pandasX.loc[0])


# Trying to get the dataset into Pytorch Dataset object
print("\n\nBEGIN PYTORCH DATASET MAKING")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AdultDataset(Dataset):
    '''Creates Adult dataset as Pytorch object from pandas dataframes'''
    def __init__(self, X_dataframe, y_dataframe):
        self.labels = y_dataframe.to_numpy()
        self.features = X_dataframe.to_numpy()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        print("\n\n__getitem__")
        print(self.features[idx])
        print(self.labels[idx])
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label
    

Adult = AdultDataset(pandasX, pandasy)
adult_loader = DataLoader(Adult, batch_size=50, shuffle=True)

print("\n\nPRINT NEW ADULT DATASET (pytorch)")
print(Adult.features)
print(Adult.labels)

print("\n\nPrinting each batch")
i = 0
for X, y in adult_loader:
    print(i)
    i += 1