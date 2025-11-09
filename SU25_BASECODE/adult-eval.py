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
from adult import Adult
from  adultsmooth import smooth_attr_bool, smooth_attr_batch, smooth_attr_num

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

# Model definitions
model_dnn_2 = nn.Sequential(
    nn.Flatten(), nn.Linear(104,200), nn.ReLU(), 
    nn.Linear(200,2)
).to(device)

model_dnn_4 = nn.Sequential(
    nn.Flatten(), nn.Linear(104,200), nn.ReLU(), 
    nn.Linear(200,100), nn.ReLU(),
    nn.Linear(100,100), nn.ReLU(),
    nn.Linear(100,2)
).to(device)

model_dnn_2.load_state_dict(torch.load("Adult-DNN2.pt", map_location=device, weights_only=True))
model_dnn_4.load_state_dict(torch.load("Adult-DNN4.pt", map_location=device, weights_only=True))


# Data
train_set = Adult(root="datasets", download=True)
test_set = Adult(root="datasets", train=False, download=True)
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
test_loader = DataLoader(test_set, batch_size=10, shuffle=False)


# Begin testing evaluation
# Probably gonna want to turn this into a function later
# also add in smoothing for multiple attributes- just smooths over sex rn


# Numeric smoothing individuals
test_results = []

for X, y in test_loader:
    for idx in range(len(X)):
        yp = smooth_attr_num(X[idx], y[idx], model_dnn_2, idx=0, n_samples=500)
        test_results.append(yp==y[idx])



test_results = []

for X, y in test_loader:
    yp = smooth_attr_batch(X, y, model_dnn_2, start_idx=58, num_att=2, n_samples=500)
    resulttemp = yp==y
    test_results.append(resulttemp)
    raise KeyboardInterrupt

test_acc = sum(test_results)/len(test_results)

print(f"Final smoothed accuracy: {test_acc}")
raise KeyboardInterrupt


# Boolean smoothing individuals
test_results = []

for X, y in test_loader:
    for idx in range(len(X)):
        yp = smooth_attr_bool(X[idx], y[idx], model_dnn_2, start_idx=58, num_att=2, n_samples=500)
        test_results.append(yp==y[idx])

test_acc = sum(test_results)/len(test_results)

print(f"Final smoothed accuracy: {test_acc}")