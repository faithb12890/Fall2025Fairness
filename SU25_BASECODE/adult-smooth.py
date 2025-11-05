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


# Inverse of the Gaussian CDF
def phi_inverse(x, mu):
    return mu + torch.sqrt(torch.tensor(2)) * torch.erfinv(2 * x - 1)

# Smoothing function from train_save_smooth
# To be adapted to smooth for Adult dataset
# All categorical data turned into True/False for each category
# Any lists are in order of how they appear in the csv's
# The Adult data from the installed module is formatted in the following way:
# Col 0: age (normalized)
# Col 1-7: workclass
#   [Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay]
# Col 8: fnlwgt (normalized)
# Col 9-24: education
#   [Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool]
# Col 25: education-num (normalized)
# Col 26-32: marital-status
#   [Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse]
# Col 33-46: occupation
#   [Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces]
# Col 47-52: relationship
#   [Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried, ]
# Col 53-57: race
#   [White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black]
# Col 58-59: sex 
#   [Female, Male]
# Col 60: capital-gain (normalized)
# Col 61: capital-loss (normalized)
# Col 62: hours-per-week (normalized)
# Col 63-103: native-country
#   [United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands]
# Col 104: income (target)
def smooth(X, y, model, sigma, n_samples=1000):
    X = X.expand(n_samples, -1, -1, -1)
    epsilon = sigma * torch.randn_like(X)
    scores = model(X + epsilon) 
    probs = torch.softmax(scores, dim=1)    
    avg_probs = probs.mean(dim=0)           
    label = torch.argmax(avg_probs)
    if label != y.item():
        radius = 0.0
        return label.item(), radius
    best_scores = torch.topk(avg_probs, 2)          
    radius = sigma * (phi_inverse(torch.tensor(best_scores.values[0].item()), 0) - phi_inverse(torch.tensor(best_scores.values[1].item()), 0)) / 2
    return label.item(), radius.item()