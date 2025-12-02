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

# Means from data (from ave_stdev_finder)
mu = torch.tensor([-1.7706e-09,  7.3888e-01,  8.2853e-02,  3.5608e-02,  3.1265e-02,
         6.8530e-02,  4.2404e-02,  4.6416e-04, -9.4855e-10,  1.6723e-01,
         2.2140e-01,  3.4746e-02,  3.2624e-01,  1.7970e-02,  3.3420e-02,
         4.3333e-02,  1.5085e-02,  1.8467e-02,  1.2499e-02,  5.3942e-02,
         5.0063e-03,  2.7187e-02,  1.2433e-02,  9.5484e-03,  1.4919e-03,
         5.0589e-09,  4.6632e-01,  1.3971e-01,  3.2246e-01,  3.1132e-02,
         2.7419e-02,  1.2267e-02,  6.9624e-04,  3.0237e-02,  1.3361e-01,
         1.0649e-01,  1.1883e-01,  1.3235e-01,  1.3388e-01,  4.4758e-02,
         6.5181e-02,  1.2337e-01,  3.2790e-02,  5.2119e-02,  4.7411e-03,
         2.1351e-02,  2.9839e-04,  4.6615e-02,  1.4807e-01,  4.1320e-01,
         2.5615e-01,  2.9474e-02,  1.0649e-01,  8.5979e-01,  2.9673e-02,
         9.4821e-03,  7.6586e-03,  9.3396e-02,  3.2432e-01,  6.7568e-01,
         1.0877e-08, -8.6002e-09, -9.8649e-09,  9.1188e-01,  5.9678e-04,
         2.8513e-03,  3.6138e-03,  3.5475e-03,  4.2438e-03,  4.6416e-04,
         3.3154e-03,  1.9561e-03,  9.6147e-04,  2.3540e-03,  2.2545e-03,
         3.0502e-03,  1.3925e-03,  3.9785e-04,  6.2330e-03,  2.2545e-03,
         1.8566e-03,  2.6523e-03,  2.1219e-03,  2.0224e-02,  1.1272e-03,
         7.9570e-04,  8.9517e-04,  2.2213e-03,  5.6362e-04,  8.9517e-04,
         1.3925e-03,  1.3925e-03,  1.8566e-03,  4.3101e-04,  2.0887e-03,
         1.0941e-03,  3.6470e-04,  5.6362e-04,  5.3047e-04,  3.3154e-03,
         5.9678e-04,  9.9463e-04,  6.2993e-04,  3.3154e-05])

# Std devs from data (from ave_stdev_finder)
sigs = torch.tensor([1.0000, 0.4393, 0.2757, 0.1853, 0.1740, 0.2527, 0.2015, 0.0215, 1.0000,
        0.3732, 0.4152, 0.1831, 0.4688, 0.1328, 0.1797, 0.2036, 0.1219, 0.1346,
        0.1111, 0.2259, 0.0706, 0.1626, 0.1108, 0.0973, 0.0386, 1.0000, 0.4989,
        0.3467, 0.4674, 0.1737, 0.1633, 0.1101, 0.0264, 0.1712, 0.3402, 0.3085,
        0.3236, 0.3389, 0.3405, 0.2068, 0.2468, 0.3289, 0.1781, 0.2223, 0.0687,
        0.1446, 0.0173, 0.2108, 0.3552, 0.4924, 0.4365, 0.1691, 0.3085, 0.3472,
        0.1697, 0.0969, 0.0872, 0.2910, 0.4681, 0.4681, 1.0000, 1.0000, 1.0000,
        0.2835, 0.0244, 0.0533, 0.0600, 0.0595, 0.0650, 0.0215, 0.0575, 0.0442,
        0.0310, 0.0485, 0.0474, 0.0551, 0.0373, 0.0199, 0.0787, 0.0474, 0.0430,
        0.0514, 0.0460, 0.1408, 0.0336, 0.0282, 0.0299, 0.0471, 0.0237, 0.0299,
        0.0373, 0.0373, 0.0430, 0.0208, 0.0457, 0.0331, 0.0191, 0.0237, 0.0230,
        0.0575, 0.0244, 0.0315, 0.0251, 0.0058])

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
    # Calc best scores and radius!!!

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


def smooth_norm_batch(X, smooth = False, sigma = 0.2, n_samples=1000):
    '''
    Function to normalize and smooth individuals.

    Arguments:
        X - batch of individuals to smooth
        model - model to be smoothed
        start_idx - first column of true/false attribute
        num_att - total number of options for attribute
        n_samples - number of samples for smoothing

    Returns:
        Xnorm - normalized batch of individuals
        Xmod - modified batch of individuals
    '''

    mu_batch = mu.expand(len(X),-1)     # Shape: [batch size, 104]
    sigs_batch = sigs.expand(len(X),-1) # Shape: [batch size, 104]
    # Selecting variables to smooth over
    smoothvars = torch.zeros(104)
    smoothvars[0] = 1       # age
    smoothvars[26:33] = 1   # marital-status
    smoothvars[53:58] = 1   # race
    smoothvars[58:60] = 1   # sex

    # Normalizing data
    Xnorm = X - mu_batch
    Xnorm = torch.div(Xnorm,sigs_batch)

    if smooth == False:
        return Xnorm
    else:
        Xmod_array = torch.tensor([])
        attributes = []
        for indiv in Xnorm:
            # Add attributes for current individual to the attribute list
            # [age, marital-status, race, sex, y, yp, rad]
            # y, yp, rad to be set later once modified individuals are processed by model
            attributes.append([indiv[0].item(), torch.argmax(indiv[26:33]).item(), torch.argmax(indiv[53:58]).item(), torch.argmax(indiv[58:60]).item(), 0, 0, 0])
            indiv = indiv.expand(n_samples,-1)  # Shape: [n_samples, 104]
            indiv = indiv.expand(1,-1,-1)       # Shape: [1, n_samples, 104]
            epsilon = (sigma*torch.rand_like(indiv))*smoothvars
            indiv = indiv + epsilon
            Xmod_array = torch.cat((Xmod_array,indiv),0)    # Shape: [batch size, n_samples, 104]  
        return Xnorm, Xmod_array, attributes