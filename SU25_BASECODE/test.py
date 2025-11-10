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


og = torch.zeros([5,5])


replace1 = torch.tensor([1,3])
replace2 = torch.tensor([[1,4,8],[12,9,3]])
replace3 = torch.tensor([[1,2],[3,4],[5,6],[7,8],[9,10]])

test1 = og.clone()
test1[1,3:5] = replace1
print(test1)

test2 = og.clone()
test2[2:4,1:4] = replace2
print(test2)

test3 = og.clone()
test3[:, 2:4] = replace3
print(test3)

print(og)

