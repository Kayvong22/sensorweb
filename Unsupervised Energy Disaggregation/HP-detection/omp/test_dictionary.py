import numpy as np
import pickle
import time
from omp.dictionary import *
import os
from experiments.test_ukdale_high.params import *

tslength = 1100
boxwidth = 1050

x = np.linspace(1, tslength, tslength)

if tslength < boxwidth:
    boxwidth = tslength

ll = []

for j in range(1, boxwidth):
    for i in range(1, tslength):
        ll.append(Boxcar(i, j, x))

X = np.array([mm for mm in ll], dtype=float).T

# # TODO centralise the paths
# if os.path.exists("/Users/pwin"):
#     path2store = '/Users/pwin/Documents/nilm_paper/nilm/_store/'

# else:
#     path2store = '/work3/s160157/'

np.save(path2store + 'X.npy', X)
