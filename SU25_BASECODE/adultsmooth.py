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
# Col 0: age (normalized to normal gaussian distribution with mean of 0, standard dev of 1)
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

def smooth_all(indiv, model, sigma=0.2, n_samples=1000):
    '''
    
    Parent smoothing function that smooths an individual across all chosen 
    protected attributes, modifying every attribute per sample. Each sample gets
    all attributes randomly perturbed.

    Arguments:
        indiv - the individual to smooth
        model - the model to be smoothed
        sigma - the sigma value to use for numeric data smoothing
        n_samples - the number of samples to generate
    '''
    # Create variable (type??) of true attributes for individual

    # Create n_samples of the individual to be smoothed
    indiv = indiv.expand(n_samples, -1)

    # Modify samples
    for i in range(n_samples):
        new_samp = smooth_samp_bool(indiv[i], 'race')
        new_samp = smooth_samp_bool(new_samp, 'sex')
        new_samp = smooth_samp_bool(new_samp, 'marital-status')
        new_samp = smooth_samp_num(new_samp, 'age', sigma=sigma)
        
        # Replace individual sample with modified versioin
        indiv[i] = new_samp
        
    # Calculate label
    scores = model(indiv) 
    probs = torch.softmax(scores, dim=1)    
    avg_probs = probs.mean(dim=0)           
    label = torch.argmax(avg_probs)

    return label.item()

def smooth_one(indiv, model, sigma=0.2, n_samples=1000):
    '''
    
    Parent smoothing function that smooths an individual across all chosen 
    protected attributes, modifying only one attribute per sample. Each sample
    gets a single randomly selected attribute randomly perturbed.

    Arguments:
        indiv - the individual to smooth
        model - the model to be smoothed
        sigma - the sigma value to use for numeric data smoothing
        n_samples - the number of samples to generate
    '''
    # Create attribute dictionary
    # Input: number generated randomly, output: list of [attribute type, attribute string]
    attr_dict = {0: ['bool','race'],
                 1: ['bool','sex'], 
                 2: ['bool','marital-status'],
                 3: ['num','age']}

    # Create variable (type??) of true attributes for individual

    # Create n_samples of the individual to be smoothed
    indiv = indiv.expand(n_samples, -1)

    # Randomly generate attributes to change. Equal probability of each.
    change_attr = torch.randint(0,len(attr_dict),(n_samples,))

    # Modify individual
    for i in range(n_samples):
        # Get corresponding attribute list, and smooth accordingly
        attr_list = attr_dict[change_attr[i].item()]
        if attr_list[0] == 'bool':
            new_samp = smooth_samp_bool(indiv[i], attr_list[1])
        elif attr_list[0] == 'num':
            new_samp = smooth_samp_num(indiv[i], attr_list[1], sigma=sigma)
        # Replace individual sample with modified versioin
        indiv[i] = new_samp
        
    # Calculate label
    scores = model(indiv) 
    probs = torch.softmax(scores, dim=1)    
    avg_probs = probs.mean(dim=0)           
    label = torch.argmax(avg_probs)

    return label.item()



# Single sample smoothing
# These functions will be called by a parent function that smooths all attributes
# One numeric and one boolean smoothing function to return a single sample

def smooth_samp_bool(sample, attribute):
    '''
    Returns a sample where the selected boolean attribute is randomly perturbed

    Arguments:
        sample - the provided sample to modify
        attribute - the selected attribute to modify, as a string used in the dictionary.
                possible inputs: 'race', 'sex', 'marital-status'

    Returns:
        mod_samp - modified sample
    '''

    # Attribute dictionary
    # Input: attribute name, output: list of [start column, number of attributes]
    attr_dict = {'race': [53,5],
                'sex': [58,2], 
                'marital-status':[26,7]}

    # Clone given sample
    mod_samp = sample.clone()

    # Get start column corresponding to selected attribute
    attr_col = attr_dict[attribute]

    # Generate attribute index to turn true
    change = torch.randint(0,attr_col[1],(1,))

    # Create replacement tensor for attribute
    replace = torch.zeros(attr_col[1])
    replace[change] = 1

    # Replace selected attribute with attribute tensor
    mod_samp[attr_col[0]:attr_col[0]+attr_col[1]] = replace

    return mod_samp


def smooth_samp_num(sample, attribute, sigma=0.2):
    '''
    Returns a sample where the selected numeric attribute is randomly perturbed

    Arguments:
        sample - the provided sample to modify
        attribute - the selected attribute to modify, as a string used in the dictionary.
                possible inputs: 'age'

    Returns:
        mod_samp - modified sample
    '''

    device = sample.device

    # Attribute dictionary
    # Input: attribute name, output: attribute column
    attr_dict = {'age':0}

    # Clone given sample
    mod_samp = sample.clone()

    # Get start column corresponding to selected attribute
    attr_col = attr_dict[attribute]

    # Generate random perturbation
    epsilon = torch.randn(1, device=device)
    epsilon = epsilon*sigma

    # Add perturbation to selected attribute
    mod_samp[attr_col] += epsilon.item()

    return mod_samp




# Individual smoothing
# These functions take in an individual and smooth over a single attribute

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

# Smoothing across numeric data

def smooth_attr_num(X, y, model, idx=0, sigma=0.2, n_samples=1000):
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
    epsilon = epsilon*sigma
    
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


# Batchwise smoothing


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