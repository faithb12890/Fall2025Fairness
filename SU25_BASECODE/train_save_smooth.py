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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

# Model definitions
model_dnn_2 = nn.Sequential(
    nn.Flatten(), nn.Linear(3072,200), nn.ReLU(), 
    nn.Linear(200,10)
).to(device)

model_dnn_4 = nn.Sequential(
    nn.Flatten(), nn.Linear(3072,200), nn.ReLU(), 
    nn.Linear(200,100), nn.ReLU(),
    nn.Linear(100,100), nn.ReLU(),
    nn.Linear(100,10)
).to(device)

model_cnn = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*64, 100), nn.ReLU(),
                          nn.Linear(100, 10)).to(device)


# Figuring out how to import uci datasets for use with pytorch

# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 
  
# metadata 
#print("\n\nMETADATA")
#print(adult.metadata) 
  
#variable information 
print("\n\nVARIABLES")
print(adult.variables)

# Changing X0:
print("\n\n old X.loc[0]")
print(X.loc[0])
newX = X.copy(deep = True)

#Update: age to 28, education to 11th, race to Asian-Pac-Islander

newX['age'] = 28
newX['education'] = '11th'
newX['race'] = 'Asian-Pac-Islander'

print("\n\n new X.loc[0]")
print(newX.loc[0])


print("\n\n Reprinting old X.loc[0]")
print(X.loc[0])

raise KeyboardInterrupt


# Data
mnist_train = datasets.CIFAR10("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.CIFAR10("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

# PGD attack parameters
training_epsilon = 0.005  # Maximum perturbation
epsilon = 0.1  # Maximum perturbation
alpha = 0.01 # Step size
num_iter = 40 # Number of iterations

# Define separate optimizers with 
opt_dnn2 = optim.SGD(model_dnn_2.parameters(), lr=0.1)
opt_dnn4 = optim.SGD(model_dnn_4.parameters(), lr=0.1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

def epoch(loader, model, opt=None):
    total_loss, total_err = 0., 0.
    for X, y in tqdm(loader, desc="Epoch Progress"):
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

# Training functions
def epoch_adversarial(loader, model, attack, opt, *args):
    total_loss, total_err = 0., 0.
    for X, y in tqdm(loader, desc="Adversarial Training"):  # Add progress bar
        X, y = X.to(device), y.to(device)
        delta = attack(model, X, y, *args)
        yp = model(X + delta)
        loss = nn.CrossEntropyLoss()(yp, y)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


# PGD L_inf Attack
def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    delta = torch.zeros_like(X, requires_grad=True)
    for _ in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()

# PGD L_2 Norm Attack
def pgd_l2(model, X, y, epsilon=2.4, alpha=0.01, num_iter=20):
    delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        
        # Update delta
        delta.data = delta.data + alpha * delta.grad.detach()
        
        # Project onto L2 ball with radius epsilon
        delta_norms = torch.norm(delta.data.view(delta.shape[0], -1), dim=1, keepdim=True)
        delta.data = delta.data / delta_norms.view(-1, 1, 1, 1) * torch.min(delta_norms, torch.tensor(epsilon).to(delta.device)).view(-1, 1, 1, 1)
        
        delta.grad.zero_()
    return delta.detach()

# Model evaluation on clean data
def evaluate_clean(model, loader):
    model.eval()
    total_err = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            total_err += (yp.max(dim=1)[1] != y).sum().item()
    return 1 - total_err / len(loader.dataset)

# Model evaluation under PGD attack
def evaluate_linf(model, loader, epsilon, alpha, num_iter):
    model.eval()
    total_err = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        delta = pgd_linf(model, X, y, epsilon, alpha, num_iter)
        yp = model(X + delta)
        total_err += (yp.max(dim=1)[1] != y).sum().item()
    return 1 - total_err / len(loader.dataset)

def evaluate_l2(model, loader, epsilon, alpha, num_iter):
    model.eval()
    total_err = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        delta = pgd_l2(model, X, y, epsilon, alpha, num_iter)
        yp = model(X + delta)
        total_err += (yp.max(dim=1)[1] != y).sum().item()
    return 1 - total_err / len(loader.dataset)

# Inverse of the Gaussian CDF
def phi_inverse(x, mu):
    return mu + torch.sqrt(torch.tensor(2)) * torch.erfinv(2 * x - 1)

# Smooth function for certified radius. Uses softmax to obtain vectors with entries in [0,1] that sum to 1 so they can be inputted into erfinv.
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

if __name__ == '__main__':
    # Train and save DNN2 if not already saved
    opt2 = optim.SGD(model_dnn_2.parameters(), lr=1e-1)
    opt4 = optim.SGD(model_dnn_4.parameters(), lr=1e-1)
    optcnn = optim.SGD(model_cnn.parameters(), lr=1e-1)

    if not os.path.exists("CIFAR10-DNN2.pt"):
        for n in range(10):
            # Fix parameter order: loader first, then model
            train_err, train_loss = epoch(train_loader, model_dnn_2, opt2)#, pgd_linf, opt_dnn2, training_epsilon, alpha, num_iter)
            train_acc = 1 - train_err
            print(f"[DNN_2] Epoch {n+1}: Train Accuracy = {train_acc:.4f}, Train Loss = {train_loss:.4f}")
        torch.save(model_dnn_2.state_dict(), "CIFAR10-DNN2.pt")

    # Train and save DNN4 if not already saved
    if not os.path.exists("CIFAR10-DNN4.pt"):
        for n in range(10):
            # Fix parameter order: loader first, then model
            train_err, train_loss = epoch(train_loader, model_dnn_4, opt4)#, pgd_linf, opt_dnn4, training_epsilon, alpha, num_iter)
            train_acc = 1 - train_err
            print(f"[DNN_4] Epoch {n+1}: Train Accuracy = {train_acc:.4f}, Train Loss = {train_loss:.4f}")
        torch.save(model_dnn_4.state_dict(), "CIFAR10-DNN4.pt")

    if not os.path.exists("CIFAR10-CNN.pt"):
        for n in range(10):
            # Fix parameter order: loader first, then model
            train_err, train_loss = epoch(train_loader, model_dnn_4, optcnn)#, pgd_linf, opt_dnn4, training_epsilon, alpha, num_iter)
            train_acc = 1 - train_err
            print(f"[CNN] Epoch {n+1}: Train Accuracy = {train_acc:.4f}, Train Loss = {train_loss:.4f}")
        torch.save(model_cnn.state_dict(), "CIFAR10-CNN.pt")


    # Loading save states
    model_dnn_2.load_state_dict(torch.load("CIFAR10-DNN2.pt", map_location=device, weights_only=True))
    model_dnn_4.load_state_dict(torch.load("CIFAR10-DNN4.pt", map_location=device, weights_only=True))
    model_cnn.load_state_dict(torch.load("CIFAR10-CNN.pt", map_location=device, weights_only=True))

    # Evaluating and printing results
    for model, name in [
        (model_dnn_2, "DNN_2"),
        (model_dnn_4, "DNN_4"),
        (model_cnn, "CNN")
    ]:
        
        clean_acc = evaluate_clean(model, test_loader)
        adv_acc = evaluate_l2(model, test_loader, epsilon, alpha, num_iter)
        print(f"Accuracy of {name} on clean data: {clean_acc:.4f}")
        print(f"Accuracy of {name} under PGD attack: {adv_acc:.4f}")

# Define ReGU activation function
def regu(x, sigma=0.1):
    # Standard normal PDF: φ(z) = exp(-z²/2) / sqrt(2π)
    phi = lambda z: torch.exp(-z**2 / 2) / math.sqrt(2 * math.pi)
    
    # Standard normal CDF: Φ(z) = (1 + erf(z/sqrt(2))) / 2
    Phi = lambda z: (1 + torch.erf(z / math.sqrt(2))) / 2
    
    regu_output = (x * Phi(x / sigma) + 
                   sigma * phi(x / sigma))
    
    return regu_output