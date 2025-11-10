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
from  adultsmooth import smooth_attr_bool, smooth_attr_batch, smooth_attr_num, smooth_all

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

# smoothing all
test_results = []
correct_rates = torch.empty((0, 3), dtype=torch.float32)

for X, y in test_loader:
    for idx in range(len(X)):
        sex = torch.argmax(X[idx][58:60])
        yp = smooth_all(X[idx], model_dnn_2, sigma=0.2, n_samples=500)
        test_results.append(yp==y[idx])
        correct_rates = torch.cat((correct_rates, torch.tensor([[sex, yp, y[idx].item()]])), dim=0)

test_acc = sum(test_results)/len(test_results)

print(f"Final smoothed accuracy: {test_acc}")

# columns:
# 0 = sex (0=male, 1=female)
# 1 = predicted label (yp)
# 2 = true label (y)

# split by sex
men_mask = correct_rates[:, 0] == 0
women_mask = correct_rates[:, 0] == 1

num_men = men_mask.sum().item()
num_women = women_mask.sum().item()

print(f"# men = {num_men}")
print(f"# women = {num_women}")

# outcomes for women
pred = correct_rates[:, 1]
true = correct_rates[:, 2]

tn_w = ((pred == 0) & (true == 0) & women_mask).sum().item()  # 00
fn_w = ((pred == 0) & (true == 1) & women_mask).sum().item()  # 01
fp_w = ((pred == 1) & (true == 0) & women_mask).sum().item()  # 10
tp_w = ((pred == 1) & (true == 1) & women_mask).sum().item()  # 11

print("Women results:")
print(f"  True Negative Women (00): {tn_w/num_women}")
print(f"  False Negative Women(01): {fn_w/num_women}")
print(f"  False Positive Women (10): {fp_w/num_women}")
print(f"  True Positive Women (11): {tp_w/num_women}")

tn_m = ((pred == 0) & (true == 0) & men_mask).sum().item()  # 00
fn_m = ((pred == 0) & (true == 1) & men_mask).sum().item()  # 01
fp_m = ((pred == 1) & (true == 0) & men_mask).sum().item()  # 10
tp_m = ((pred == 1) & (true == 1) & men_mask).sum().item()  # 11

print("Men results:")
print(f"  True Negative Men (00): {tn_m/num_men}")
print(f"  False Negative Men(01): {fn_m/num_men}")
print(f"  False Positive Men (10): {fp_m/num_men}")
print(f"  True Positive Men (11): {tp_m/num_men}")

raise KeyboardInterrupt

test_results = []

for X, y in test_loader:
    yp = smooth_attr_batch(X, y, model_dnn_2, start_idx=58, num_att=2, n_samples=500)
    resulttemp = yp==y
    test_results.append(resulttemp)
    raise KeyboardInterrupt

test_acc = sum(test_results)/len(test_results)

print(f"Final smoothed accuracy: {test_acc}")



# Boolean smoothing individuals
test_results = []

for X, y in test_loader:
    for idx in range(len(X)):
        yp = smooth_attr_bool(X[idx], y[idx], model_dnn_2, start_idx=58, num_att=2, n_samples=500)
        test_results.append(yp==y[idx])

test_acc = sum(test_results)/len(test_results)

print(f"Final smoothed accuracy: {test_acc}")