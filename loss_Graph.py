import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import mse_loss
import collections
import matplotlib
import matplotlib.pyplot as plt
import random
import math
from torch.autograd import Variable
import glob
import os
import torch.nn.functional as F
import copy
from tqdm import tqdm
import datetime
import scipy.io as sio
import pickle

with open ("loss_sum_8.pkl","rb") as f:
    loss_sum = pickle.load(f) #8-0.0001,9-0.001
print(loss_sum)
x = np.arange(50000)
plt.plot(x,loss_sum)
plt.xlabel("Epochs")
plt.ylabel("Train_loss")
plt.show()