import pandas as pd
from ucimlrepo import fetch_ucirepo
from adult import Adult

adult = fetch_ucirepo("adult")
X = adult.data.features

# Select numeric columns
numeric_df = X.select_dtypes(include=["int64", "float64"])

means = numeric_df.mean()
stds = numeric_df.std()

print("Column means:")
print(means)

print("\nColumn standard deviations:")
print(stds)

import torch
from torch.utils.data import DataLoader

train_set = Adult(root="datasets", download=True)
train_loader = DataLoader(train_set, batch_size=2048, shuffle=False)

all_rows = []

for X_batch, _ in train_loader:
    # convert bools to float (True→1, False→0)
    X_batch = X_batch.float()
    all_rows.append(X_batch)

# Combine all batches into one (N, 104) tensor
X_all = torch.cat(all_rows, dim=0)

# Column-wise mean and std
col_means = X_all.mean(dim=0)   # shape: (104,)
col_stds  = X_all.std(dim=0)

print("Means:", col_means)
print("Stds:", col_stds)
