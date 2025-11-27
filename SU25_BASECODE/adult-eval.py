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

#(liv Runs)
#model_dnn_2.load_state_dict(torch.load("SU25_BASECODE/models/Adult-DNN2.pt", map_location=device, weights_only=True))
#model_dnn_4.load_state_dict(torch.load("SU25_BASECODE/models/Adult-DNN4.pt", map_location=device, weights_only=True))

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
        yp = smooth_all(X[idx], model_dnn_2, sigma=0.2, n_samples=5)
        test_results.append(yp==y[idx])
        correct_rates = torch.cat((correct_rates, torch.tensor([[sex, yp, y[idx].item()]])), dim=0)

test_acc = sum(test_results)/len(test_results)

print(f"Final smoothed accuracy: {test_acc}")

def _confusion_counts(mask, pred, true):
    """Return tn, fp, fn, tp, n for a boolean mask."""
    g_pred = pred[mask]
    g_true = true[mask]
    tn = ((g_pred == 0) & (g_true == 0)).sum().item()
    fp = ((g_pred == 1) & (g_true == 0)).sum().item()
    fn = ((g_pred == 0) & (g_true == 1)).sum().item()
    tp = ((g_pred == 1) & (g_true == 1)).sum().item()
    n = tn + fp + fn + tp
    return tn, fp, fn, tp, n


def report_by_binary_attribute(correct_rates, attr_col=0, group_names=("Group 0", "Group 1")):
    """
    correct_rates: tensor with columns:
        0 = attribute (0 or 1)
        1 = predicted label
        2 = true label
    """
    attr = correct_rates[:, attr_col]
    pred = correct_rates[:, 1]
    true = correct_rates[:, 2]
    for value, name in enumerate(group_names):
        mask = (attr == value)
        if mask.sum().item() == 0:
            print(f"{name}: no samples")
            continue
        tn, fp, fn, tp, n = _confusion_counts(mask, pred, true)
        print(f"{name} results (N = {n}):")
        print(f"  True Negative (00): {tn / n:.4f}")
        print(f"  False Negative (01): {fn / n:.4f}")
        print(f"  False Positive (10): {fp / n:.4f}")
        print(f"  True Positive (11): {tp / n:.4f}")
        print()


def fairness_metrics(correct_rates, attr_col=0):
    """
    Computes standard group fairness metrics.
    Returns:
        metrics: {group_value: {TPR, FPR, FNR, PPR, Accuracy}}
        diffs:   {metric_name: |metric_group0 - metric_group1|}
    """
    attr = correct_rates[:, attr_col]
    pred = correct_rates[:, 1]
    true = correct_rates[:, 2]
    metrics = {}
    for group in [0, 1]:
        mask = (attr == group)
        tn, fp, fn, tp, n = _confusion_counts(mask, pred, true)
        if n == 0:
            metrics[group] = None
            continue
        tpr = tp / (tp + fn + 1e-8)          # True Positive Rate
        fpr = fp / (fp + tn + 1e-8)          # False Positive Rate
        fnr = fn / (fn + tp + 1e-8)          # False Negative Rate
        ppr = (tp + fp) / (n + 1e-8)         # Positive Prediction Rate
        acc = (tp + tn) / (n + 1e-8)         # Accuracy
        metrics[group] = {
            "TPR": tpr,
            "FPR": fpr,
            "FNR": fnr,
            "PPR": ppr,
            "Accuracy": acc,
        }

    if metrics[0] is not None and metrics[1] is not None:
        diffs = {m: abs(metrics[0][m] - metrics[1][m]) for m in metrics[0]}
    else:
        diffs = None
    return metrics, diffs


report_by_binary_attribute(correct_rates,attr_col=0,group_names=("Women", "Men") ) # group 0 == women, 1 == men)

metrics, diffs = fairness_metrics(correct_rates)

print("Men metrics:", metrics[0])
print("Women metrics:", metrics[1])
print("Differences:", diffs)

metrics, diffs = fairness_metrics(correct_rates)

print("Men metrics:", metrics[0])
print("Women metrics:", metrics[1])
print("Differences:", diffs)

#Accuracy sanity check 
attr = correct_rates[:, 0]
n0 = (attr == 0).sum().item()
n1 = (attr == 1).sum().item()
N = n0 + n1

acc0 = metrics[0]["Accuracy"]
acc1 = metrics[1]["Accuracy"]

weighted_acc = (acc0 * n0 + acc1 * n1) / N

print(f"\nSanity check:")
print(f"  Overall test_acc        = {test_acc:.6f}")
print(f"  Weighted group accuracy = {weighted_acc:.6f}")
print(f"  Difference              = {abs(test_acc - weighted_acc):.6e}")


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