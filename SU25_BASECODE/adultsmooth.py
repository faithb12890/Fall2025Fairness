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

# We want to smooth over sex (col 58-59), race (col 53-57) and age (col 0). These are the noted protected attributes.
# We may also be able to smooth over marital-status (col 26-32) as other fairness sets note this as a protected attribute.
# We can potentially create sub-smoothing functions that will smooth individual attributes, and a separate smoothing function that will randomly select a subfunctiont to use.

def smooth_attr_bool(X, y, model, start_idx = 58, num_att = 2, n_samples=1000):
    '''
    Function to smooth over a single feature in the Adult dataset. Smooths an
    individual, not a batch. Smooths over sex by default.

    Note (Faith): May work out how to smooth over batches later- need to replace
    certain parts of X with a replacement vector. This is easier for now, but may
    be slower.

    Arguments:
        X - single individual to smooth
        y - target attribute
        model - model to be smoothed
        start_idx - first column of true/false attribute
        num_att - total number of options for attribute
        n_samples - number of samples for smoothing
    '''
    X = X.expand(n_samples, -1)
    end = start_idx+num_att # First column that isn't related to the categorical variable to be smoothed

    # Randomly generate smoothed attribute
    change = torch.randint(0,num_att,(n_samples,))
    
    # Build smoothed data
    for i in range(len(X)):
        # Create replacement tensor
        replace = torch.zeros(num_att)
        # Turn the randomly selected category true
        replace[change[i]] = 1
        # Replace all values in categorical attribute with replacement tensor
        X[i][start_idx:end] = replace

    scores = model(X) 
    probs = torch.softmax(scores, dim=1)    
    avg_probs = probs.mean(dim=0)           
    label = torch.argmax(avg_probs)
    # Need to replace radius calcs
    #if label != y.item():
        #radius = 0.0
        #return label.item(), radius
    #best_scores = torch.topk(avg_probs, 2)          
    #radius = sigma * (phi_inverse(torch.tensor(best_scores.values[0].item()), 0) - phi_inverse(torch.tensor(best_scores.values[1].item()), 0)) / 2
    return label.item()

def smooth_attr_batch(X, y, model, start_idx = 58, num_att = 2, n_samples=1000):
    '''
    INCOMPLETE

    Function to smooth over a single feature in the Adult dataset. Smooths a
    batch. Smooths over sex by default.

    Arguments:
        X - single individual to smooth
        y - target attribute
        model - model to be smoothed
        start_idx - first column of true/false attribute
        num_att - total number of options for attribute
        n_samples - number of samples for smoothing
    '''
    X = X.expand(n_samples, -1, -1)
    end = start_idx+num_att # First column that isn't related to the categorical variable to be smoothed

    # Randomly generate smoothed attribute
    change = torch.randint(0,num_att,(n_samples,10))

    # Create replacement tensor 
    replace = torch.zeros(n_samples,len(X[0]),num_att)

    # Turn selected attribute in replacement tensor true
    for i in range(len(change)):
        for j in range(len(change[i])):
            replace[i][j][change[i,j]] = 1

    # Replace attributes with smoothed versions
    smoothX = X.clone()
    smoothX[:,:, start_idx:end] = replace
    print(smoothX.size())

    scores = model(X) 
    probs = torch.softmax(scores, dim=1)    
    avg_probs = probs.mean(dim=0)           
    label = torch.argmax(avg_probs)
    print(label)
    print(label.size())
    # Need to replace radius calcs
    #if label != y.item():
        #radius = 0.0
        #return label.item(), radius
    #best_scores = torch.topk(avg_probs, 2)          
    #radius = sigma * (phi_inverse(torch.tensor(best_scores.values[0].item()), 0) - phi_inverse(torch.tensor(best_scores.values[1].item()), 0)) / 2
    return label.item()



# Smoothing across numeric data

def smooth_attr_num(X, y, model, idx = 0, n_samples=1000):
    '''
    Function to smooth over a single feature in the Adult dataset. Smooths an
    individual, not a batch. Smooths over sex by default.

    Note (Faith): May work out how to smooth over batches later- need to replace
    certain parts of X with a replacement vector. This is easier for now, but may
    be slower.

    Arguments:
        X - single individual to smooth
        y - target attribute
        model - model to be smoothed
        idx - column of numeric data to be smoothed
        n_samples - number of samples for smoothing
    '''

    device = X.device
    X = X.expand(n_samples, -1)

    epsilon = torch.randn(n_samples, device=device)
    print(epsilon)
    
    newX = X.clone()
    newX[:,0] = newX[:,0] + epsilon

    scores = model(X) 
    probs = torch.softmax(scores, dim=1)    
    avg_probs = probs.mean(dim=0)           
    label = torch.argmax(avg_probs)
    # Need to replace radius calcs
    #if label != y.item():
        #radius = 0.0
        #return label.item(), radius
    #best_scores = torch.topk(avg_probs, 2)          
    #radius = sigma * (phi_inverse(torch.tensor(best_scores.values[0].item()), 0) - phi_inverse(torch.tensor(best_scores.values[1].item()), 0)) / 2
    return label.item()