#! usr/bin/env python3
# %%
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
import math
import re
import time
import networkx as nx
import matplotlib.cm as cm
import dill
import scipy.optimize as sci
from multiprocessing import Pool
import pickle
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from sklearn.mixture import GaussianMixture
import seaborn as sns
sns.set()
from matplotlib.patches import Ellipse
from statistics import median
from tslearn.metrics import dtw, soft_dtw
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


import dill
def save_pickle(obj,file_name):
    with open(file_name, 'wb') as handle:
        dill.dump(obj, handle, protocol=dill.HIGHEST_PROTOCOL)
    return file_name + ' is saved'

def open_pickle(file_name):
    with open(file_name, 'rb') as handle:
        obj = dill.load(handle)
    return obj
# %%
washerdryer = open_pickle('/Users/kayvon/Desktop/washerdryer.pkl')
dishwasher = open_pickle('/Users/kayvon/Desktop/dishwasher.pkl')
fridgefreezer = open_pickle('/Users/kayvon/Desktop/fridgefreezer.pkl')
kettle = open_pickle('/Users/kayvon/Desktop/kettle.pkl')
microwave = open_pickle('/Users/kayvon/Desktop/microwave.pkl')

dictionary = open_pickle('/Users/kayvon/Desktop/dictionary2000.pkl')

washerdryer = np.array(washerdryer)
dishwasher = np.array(dishwasher)
fridgefreezer = np.array(fridgefreezer)
kettle = np.array(kettle)
microwave = np.array(microwave)

appliance_dict = {}
appliance_dict['washerdryer'] = washerdryer
appliance_dict['dishwasher'] = dishwasher
appliance_dict['fridgefreezer'] = fridgefreezer
appliance_dict['kettle'] = kettle
appliance_dict['microwave'] = microwave

# dictionary_df = pd.DataFrame(dictionary)

# dictionary_df = dictionary_df.T
# dictionary_df = dictionary_df.reindex(index=dictionary_df.index[::-1])

# dictionary_df = dictionary_df.reset_index(drop=True)
# dictionary_df = dictionary_df.T

# dictionary2 = dictionary_df.to_numpy(dtype=float)
# %%
def norm2(x):
    """Compute a l2-norm."""
    return np.linalg.norm(x) / np.sqrt(len(x))

class Result():
    """Aggregate the result from the OMP algorithm.


    Args:
        -- dataframe : df with all the individual appliances sample
        -- signal : simply the full input signal;
        -- maxiter : number maximum of iteration befroe the algorithm stops;
        -- ztol : error tolerance;
        -- S : sparsity threshold (needs to be deduce empirically);
        -- coef_select : nonzero coefficients in "signal =~ coef * Dictionary";
        -- xmax : maximum value of the signal;
        -- k_select : K strongest atoms;
        -- RecSignal : reconstituted signal;
        -- err : relative l2 error;
        -- D : Dictionary;
        -- k_index_nonzero : index in the dictionary for the atoms with
                                nonzero coefficents.

    """

    def __init__(self, **kwargs):

        # Input
        self.dataframe = None
        self.signal = None
        self.maxiter = None
        self.tol = None
        self.ztol = None
        self.S = None

        # Output
        self.coef_select = None
        self.xmax = None
        self.k_select = None
        self.RecSignal = None
        self.err = None
        self.resCovk = None

        # Combined output
        self.Kcoef = None
        self.nbatoms = None
        self.full_y = None

        # Cluster GMM
        self.REFIND = None
        # self.optGMMcomponents = None

        # Community DETECTION
        self.ComDict = None
        self.Graph = None
        self.partition = None

        # After labelling process
        self.y_hat_dict = None
        self.y_truth_dict = None

    def update(self, dataframe, signal,
               maxiter, tol, ztol, S,
               coef_select, xmax, k_select, RecSignal, k_index_nonzero,
               err, resCovk,
               Kcoef, nbatoms, full_y,
               REFIND, optGMMcomponents,
               ComDict, Graph, partition,
               y_hat_dict, y_truth_dict):
        '''Update the solution attributes.
        '''

        self.dataframe = dataframe
        self.signal = signal
        self.maxiter = maxiter
        self.tol = tol
        self.ztol = ztol
        self.S = S
        self.coef_select = coef_select
        self.xmax = xmax
        self.k_select = k_select
        self.RecSignal = RecSignal
        self.err = err
        self.resCovk = resCovk
        self.k_index_nonzero = k_index_nonzero

        self.Kcoef = Kcoef
        self.nbatoms = nbatoms
        self.full_y = full_y

        self.REFIND = REFIND
        self.optGMMcomponents = optGMMcomponents

        self.ComDict = ComDict
        self.Graph = Graph
        self.partition = partition

        self.y_hat_dict = y_hat_dict
        self.y_truth_dict = y_truth_dict

def rlencode(x, dropna=False):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle
    function from R.

    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropna: bool, optional
        Drop all runs of NaNs.

    Returns
    -------
    start positions, run lengths, run values

    """
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=x.dtype))

    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    
    values = x[starts]


    if dropna:
        mask = ~np.isnan(values)
        starts, lengths, values = starts[mask], lengths[mask], values[mask]

    return starts, lengths, values

def OMP(x, Dictionary, dataframe,
        maxiter=1000, ztol=1e-12, tol=1e-10, S=100000,
        threshold_min_power=0):
    """Compute the Orthogonal Matching Pursuit.

    Arguments:
        x -- aggregated load signal
        Dictionary -- Large collection of atoms, each column is an atom
        segment -- Array [start, stop]
        maxiter -- Maximum number of iteration, each iteration the algorithm
                   select an atom 'kj' and update the residual 'Rj'
                   (default=100).
        S -- Sparsity convoyed through a maximum number of coefficient
                (default=1000)
        ztol -- tolerance on the maximum residual covariance allowed,
                i.e. iterations breaks if threshold is achieved
        tol -- convergence tolerance, breaks if the relative error is than
                tol * norm2(x)


    Returns:
        Set of active atoms 'k' with their respective coefficient 'coef'
    """
    # -------------- Check structure args -------------------------------------

    if not type(x).__module__ == np.__name__:  # ifnot numpy matrix
        x = x.values
        x = x[:, 0]  # denesting ndarray

    # -------------- Initialization -------------------------------------------

    xmax = np.max(x)                            # normalizing the signal
    x = x / xmax

    D = Dictionary

    k_index = []                                # vector for the selected atoms
    coef = np.zeros(D.shape[1], dtype=float)    # coefficient vector
    R = x                                       # residual vector for j=1

    xnorm = norm2(x)                            # compute relative err
    err = np.zeros(maxiter, dtype=float)        # relative err vector

    result = Result()

    # ------------- Main interation -------------------------------------------

    for j in range(maxiter):
        # Equation (3) in Arberet et al.
        Rcov = np.dot(D.T, R)       # p-correlation with residual at j=it
        k = np.argmax(Rcov)         # atom indices maximizing p-correlation

        if k not in k_index:        # if the selected atom is not already
            k_index.append(k)       # in the vector, then append it

        # Equation (2) in Arberet et al.
        coefi, _ = sci.nnls(D[:, k_index], x)   # non-negative l.s. solver
        coef[k_index] = coefi

        R = x - np.dot(D[:, k_index], coefi)    # new residual computed
        err[j] = norm2(R)                       # errors for each iteration
        #print(j)

        # TODO Delete certain type of thresholds
        # Stopping criteria :
        resCovk = Rcov[k]
        if resCovk < ztol:
            # print('Stopping criteria: all residual covariances below the threshold')
            break

        if err[j] < tol:
            # print('Stopping criteria: Convergence tolerance achieved')
            break

        if len(k_index) >= S:
            # print('\nLimit on selected atoms achieved')
            break

    # ------------- Additional Atom Selection ---------------------------------

    # Remove the zero coefficients from the support vector
    coef_select = coef[k_index]
    List = list(np.nonzero(coef_select)[0])
    k_index_nonzero = [k_index[i] for i in List]

    # Remove K strongest atom
    k_index_select = []

    for i, ll in enumerate(k_index_nonzero):
        val_rle_output = rlencode(D[:, ll] * coef[ll] * xmax)[2]

        if threshold_min_power < val_rle_output.max():
            k_index_select.append(ll)
        else:
            continue

    # ------------- Preparing Result Outputs ----------------------------------

    signal = x * xmax

    # -------- Output (supplementary) -------- #

    coef_select = coef[k_index_select]
    k_select = D[:, k_index_select]
    RecSignal = np.sum(k_select * coef_select, axis=1) * xmax

    # COMBINED output
    Kcoef = k_select * coef_select * xmax
    nbatoms = k_select.shape[1]
    full_y = signal

    # TO DO clear this and delete update ... 

    REFIND = None  # free space for the labels from the clustering method
    optGMMcomponents = None
    ComDict = None  # free space for community detection
    Graph = None
    partition = None

    y_hat_dict = None
    y_truth_dict = None

    # RESULT output update
    result.update(dataframe, signal,
                  maxiter, tol, ztol, S,
                  coef_select, xmax, k_select, RecSignal, k_index_select,
                  err, resCovk,
                  Kcoef, nbatoms, full_y,
                  REFIND, optGMMcomponents,
                  ComDict, Graph, partition,
                  y_hat_dict, y_truth_dict)

    return result

def Standardize(x):
    mean = np.mean(x)
    standard_deviation = np.std(x)

    return (x - mean) / standard_deviation
# %%
washerdryer_smooth = np.pad(washerdryer, (0, (2000-len(washerdryer))), 'constant', constant_values=(0))
dishwasher_smooth = np.pad(dishwasher, (0, (2000-len(dishwasher))), 'constant', constant_values=(0))
fridgefreezer_smooth = np.pad(fridgefreezer, (0, (2000-len(fridgefreezer))), 'constant', constant_values=(0))
kettle_smooth = np.pad(kettle, (0, (2000-len(kettle))), 'constant', constant_values=(0))
microwave_smooth = np.pad(microwave, (0, (2000-len(microwave))), 'constant', constant_values=(0))

omp_dict = {}
omp_dict['washerdryer'] = washerdryer_smooth
omp_dict['dishwasher'] =  dishwasher_smooth
omp_dict['fridgefreezer'] =  fridgefreezer_smooth
omp_dict['kettle'] = kettle_smooth
omp_dict['microwave'] = microwave_smooth

result = {}
for i in omp_dict.keys():
    df = pd.DataFrame()
    result[i] = OMP(x=omp_dict[i], Dictionary=dictionary, dataframe=df, maxiter=5000, tol=0.055, S=200, threshold_min_power=0)

washerdryer_omp = result['washerdryer'].RecSignal[0:len(appliance_dict['washerdryer'])]
dishwasher_omp = result['dishwasher'].RecSignal[0:len(appliance_dict['dishwasher'])]
fridgefreezer_omp = result['fridgefreezer'].RecSignal[0:len(appliance_dict['fridgefreezer'])]
kettle_omp = result['kettle'].RecSignal[0:len(appliance_dict['kettle'])]
microwave_omp = result['microwave'].RecSignal[0:len(appliance_dict['microwave'])]
# %%
half_washer = np.zeros((result['washerdryer'].Kcoef.shape[1],result['washerdryer'].Kcoef.shape[0],1))
for i in range(result['washerdryer'].Kcoef.shape[1]):
    half_washer[i,:,0] = result['washerdryer'].Kcoef[:,i]
# half_washer += result['washerdryer'].Kcoef
# %%
dba_km = TimeSeriesKMeans(n_clusters=2,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10)
y_pred = dba_km.fit_predict(half_washer)
# %%
washer_labels = dba_km.labels_
summed_washer_clusters = {}
for g in np.unique(washer_labels):
    summed_washer_clusters[g] = np.zeros(half_washer.shape[1])
    for l in np.where(washer_labels == g):
        for i in range(len(l)):
            temp = half_washer[l[i],:,0]
            summed_washer_clusters[g] += temp
# %%
cdict = {0: 'red', 1: 'blue', 2: 'green'}
for i in range(len(half_washer)):
    plt.plot(half_washer[i,:,0], label=washer_labels[i], color=cdict[washer_labels[i]])#'blue' if labels[i] == 1 else 'green')
    plt.ylim([0,4000])
    plt.legend()

# %%
noisy = {}
noisy['washerdryer'] = np.array(washerdryer)
noisy['dishwasher'] = np.array(dishwasher)
noisy['fridgefreezer'] = np.array(fridgefreezer)
noisy['kettle'] = np.array(kettle)
noisy['microwave'] = np.array(microwave)

omp_smooth = {}
omp_smooth['washerdryer'] = washerdryer_omp
omp_smooth['dishwasher'] =  dishwasher_omp
omp_smooth['fridgefreezer'] =  fridgefreezer_omp
omp_smooth['kettle'] = kettle_omp
omp_smooth['microwave'] = microwave_omp
#%%
########## NOISY AGGREGATE ###########
col = ['washerdryer','dishwasher','fridgefreezer','kettle','microwave']
data = np.zeros((2,len(col)),dtype=int)
agg_actual = pd.DataFrame(data =data,columns=col)

agg =  np.zeros(dictionary.shape[0])

for i in noisy.keys():
    num = np.random.randint(2000-len(noisy[i]))
    if i == 'washerdryer':
        num = np.random.randint(2000-len(noisy[i]))
        agg[num:num+noisy[i].shape[0]] += noisy[i]
        print('Washer Dryer placement = ' + str(num))
        agg_actual[i][0] = num
    elif i == 'dishwasher':
        num = np.random.randint(2000-len(noisy[i]))
        agg[num:num+noisy[i].shape[0]] += noisy[i]
        print('Dishwasher placement = ' + str(num))
        agg_actual[i][0] = num
    elif i == 'fridgefreezer':
        for j in range(2):
            num = np.random.randint(2000-len(noisy[i]))
            agg[num:num+noisy[i].shape[0]] += noisy[i]
            print('Fridge Freezer placement = ' + str(num))
            agg_actual[i][j] = num
    elif i == 'kettle':
        for k in range(2):
            num = np.random.randint(2000-len(noisy[i]))
            agg[num:num+noisy[i].shape[0]] += noisy[i]
            print('Kettle placement = ' + str(num))
            agg_actual[i][k] = num
    elif i == 'microwave':
        for z in range(2):
            num = np.random.randint(2000-len(noisy[i]))
            agg[num:num+noisy[i].shape[0]] += noisy[i]
            print('Microwave placement = ' + str(num))
            agg_actual[i][z] = num


########### PLOT AGGREGATE ############
plt.plot(agg)
#### TABLE FOR APPLIANCE PLACEMENT ####
agg_actual
# %%
col = ['washerdryer','dishwasher','fridgefreezer','kettle','microwave']
data = np.zeros((2,len(col)),dtype=int)
agg_actual = pd.DataFrame(data =data,columns=col)

########## SMOOTH AGGREGATE ###########
agg =  np.zeros(dictionary.shape[0])

for i in omp_smooth.keys():
    num = np.random.randint(2000-len(omp_smooth[i]))
    if i == 'washerdryer':
        num = np.random.randint(2000-len(omp_smooth[i]))
        agg[num:num+omp_smooth[i].shape[0]] += omp_smooth[i]
        print('Washer Dryer placement = ' + str(num))
        agg_actual[i][0] = num
    elif i == 'dishwasher':
        num = np.random.randint(2000-len(omp_smooth[i]))
        agg[num:num+omp_smooth[i].shape[0]] += omp_smooth[i]
        print('Dishwasher placement = ' + str(num))
        agg_actual[i][0] = num
    elif i == 'fridgefreezer':
        for j in range(2):
            num = np.random.randint(2000-len(omp_smooth[i]))
            agg[num:num+omp_smooth[i].shape[0]] += omp_smooth[i]
            print('Fridge Freezer placement = ' + str(num))
            agg_actual[i][j] = num
    elif i == 'kettle':
        for k in range(2):
            num = np.random.randint(2000-len(omp_smooth[i]))
            agg[num:num+omp_smooth[i].shape[0]] += omp_smooth[i]
            print('Kettle placement = ' + str(num))
            agg_actual[i][k] = num
    elif i == 'microwave':
        for z in range(2):
            num = np.random.randint(2000-len(omp_smooth[i]))
            agg[num:num+omp_smooth[i].shape[0]] += omp_smooth[i]
            print('Microwave placement = ' + str(num))
            agg_actual[i][z] = num

########### PLOT AGGREGATE ############
plt.plot(agg)
#### TABLE FOR APPLIANCE PLACEMENT ####
agg_actual
# %%
save_pickle(agg, '/Users/kayvon/Desktop/agg4.pkl')
# %%
df2 = pd.DataFrame()
smooth_result = OMP(x=agg, Dictionary=dictionary, dataframe=df2, maxiter=1000, tol=0.02, S=50, threshold_min_power=0)
# %%
df1 = pd.DataFrame()
# result_final = OMP(x=newagg, Dictionary=dictionary, dataframe=df1, maxiter=1000, tol=0.002, S=16, threshold_min_power=0)
# %%
########## ACTUAL METHOD ##########
df1 = pd.DataFrame()
result_final = OMP(x=agg, Dictionary=dictionary, dataframe=df1, maxiter=1000, tol=0.002, S=30, threshold_min_power=0)
# %%
plt.plot(result_final.Kcoef);
# %%
# result_final = gamma.dot(dictionary)
# %%
# Before GMM compare each boxcar to each signature and then filter out fridgefreezer,
# kettle, and microwave. Then with these type 1 appliances taken out do the GMM on the rest
kettle_index_list = []
for i in range(result_final.Kcoef.shape[1]):
    # dtw_score_kettle = dtw(result_final.Kcoef[:,i], kettle_omp)
    # print('DTW Kettle Score ' + str(i) + '= ' +str(dtw_score_kettle))
    
    ind = np.where(result_final.Kcoef[:,i] != 0)
    start = ind[0][0] - 2
    end = ind[0][-1] + 2
    if (len(kettle_omp) - (0.5*len(kettle_omp))) <= (end - start) <= (len(kettle_omp) + (0.5*len(kettle_omp))):
        # distance_score_kettle = np.linalg.norm(result_final.Kcoef[start:end,i] - kettle_omp)
        dtw_score_kettle = dtw(result_final.Kcoef[:,i], kettle_omp)
        print('Kettle DTW Score ' + str(i) + '= ' +str(dtw_score_kettle))
        if dtw_score_kettle <= 500:
            kettle_index_list.append(i)
    else:
        continue

    # if dtw_score_kettle < 5:
    #     kettle_index_list.append(i)
    
kettle_index_list
# %%

micro_index_list = []
for i in range(result_final.Kcoef.shape[1]):
    # dtw_score_kettle = dtw(result_final.Kcoef[:,i], kettle_omp)
    # print('DTW Kettle Score ' + str(i) + '= ' +str(dtw_score_kettle))
    
    ind = np.where(result_final.Kcoef[:,i] != 0)
    start = ind[0][0] - 2
    end = ind[0][-1] + 2
    if (len(microwave_omp) - (0.5*len(microwave_omp))) <= (end - start) <= (len(microwave_omp) + (0.5*len(microwave_omp))):
        # distance_score_micro = np.linalg.norm(result_final.Kcoef[start:end,i] - microwave_omp)
        dtw_score_micro = dtw(result_final.Kcoef[:,i], microwave_omp)
        print('Micro DTW Score ' + str(i) + '= ' +str(dtw_score_micro))
        if dtw_score_micro <= 300:
            micro_index_list.append(i)
    else:
        continue

    # if dtw_score_kettle < 5:
    #     kettle_index_list.append(i)
    
micro_index_list
# %%

fridge_index_list = []
for i in range(result_final.Kcoef.shape[1]):
    # dtw_score_kettle = dtw(result_final.Kcoef[:,i], kettle_omp)
    # print('DTW Kettle Score ' + str(i) + '= ' +str(dtw_score_kettle))
    
    ind = np.where(result_final.Kcoef[:,i] != 0)
    start = ind[0][0] - 2
    end = ind[0][-1] + 2
    # if (max(fridgefreezer_omp) - 10) <= max(result_final.Kcoef[:,i]) <= (max(fridgefreezer_omp) + 10):
    if (len(fridgefreezer_omp) - (0.5*len(fridgefreezer_omp))) <= (end - start) <= (len(fridgefreezer_omp) + (0.5*len(fridgefreezer_omp))):
        # distance_score_micro = np.linalg.norm(result_final.Kcoef[start:end,i] - microwave_omp)
        dtw_score_fridge = dtw(result_final.Kcoef[:,i], fridgefreezer_omp)
        print('Fridge DTW Score ' + str(i) + '= ' +str(dtw_score_fridge))
        if dtw_score_fridge <= 100:
            fridge_index_list.append(i)
    else:
        continue

    # if dtw_score_kettle < 5:
    #     kettle_index_list.append(i)
    
fridge_index_list

# %%
kettles_index_list = []
for i in range(result_final.Kcoef.shape[1]):
    dtw_score_kettle = dtw(result_final.Kcoef[:,i], kettle_omp)
    print('Kettle DTW Score ' + str(i) + '= ' +str(dtw_score_kettle))
    if dtw_score_kettle < 200:
        kettles_index_list.append(i)
kettles_index_list
# %%
microwave_index_list = []
for i in range(result_final.Kcoef.shape[1]):
    dtw_score_microwave = dtw(result_final.Kcoef[:,i], microwave_omp)
    print('Microwave DTW Score ' + str(i) + '= ' +str(dtw_score_microwave))
    if dtw_score_microwave < 200:
        microwave_index_list.append(i)
microwave_index_list
# %%
fridgefreezer_index_list = []
for i in range(result_final.Kcoef.shape[1]):
    dtw_score_fridgefreezer = dtw(result_final.Kcoef[:,i], fridgefreezer_omp)
    print('Fridge Freezer DTW Score ' + str(i) + '= ' +str(dtw_score_fridgefreezer))
    if dtw_score_fridgefreezer < 200:
        fridgefreezer_index_list.append(i)
fridgefreezer_index_list
# %%
fig, ax = plt.subplots()

for j in kettles_index_list:
    ax.plot(result_final.Kcoef[:,j], color='blue', 
    label='Kettle' if j == kettles_index_list[0] else "")
for i in microwave_index_list:
    ax.plot(result_final.Kcoef[:,i], color='red', 
    label='Microwave' if i == microwave_index_list[0] else "")
for k in fridgefreezer_index_list:
    ax.plot(result_final.Kcoef[:,k], color='green', 
    label='Fridgefreezer' if k == fridgefreezer_index_list[0] else "")

plt.ylim([0, 4000])
plt.legend()
plt.show()
# %%
indexes = np.arange(0,result_final.Kcoef.shape[1]).tolist()
type1_indexes = []
for i in range(len(kettles_index_list)):
    type1_indexes.append(kettles_index_list[i])
for j in range(len(microwave_index_list)):
    type1_indexes.append(microwave_index_list[j])
for k in range(len(fridgefreezer_index_list)):
    type1_indexes.append(fridgefreezer_index_list[k])

# %%
for element in type1_indexes:
    if element in indexes:
        indexes.remove(element)
# %%
fig, ax = plt.subplots()
for i in indexes:
    ax.plot(result_final.Kcoef[:,i])

plt.ylim([0,4000])
plt.show()
# %%
leftover = np.zeros(2000)
for i in indexes:
    temp = result_final.Kcoef[:,i]
    leftover += temp

fig, ax = plt.subplots()

ax.plot(leftover, label='Leftover')
plt.ylim([0,4000])
plt.legend()
plt.show()

# %%
# fig, ax = plt.subplots()

# ax.plot(result_final.Kcoef)
# plt.ylim([0,4000])
# plt.show()
# %%
dataset_small = np.zeros((result_final.Kcoef.shape[1],result_final.Kcoef.shape[0],1))

for i in indexes:
    dataset_small[i,:,0] = result_final.Kcoef[:,i]

# for i in type1_indexes:
dataset_small = np.delete(dataset_small, type1_indexes, axis=0)

dba_km1 = TimeSeriesKMeans(n_clusters=3,
                          n_init=3,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10)
y_pred = dba_km1.fit_predict(dataset_small)

labels = dba_km1.labels_

cdict = {0: 'red', 1: 'blue', 2: 'green'}
for i in range(len(dataset_small)):
    plt.plot(dataset_small[i,:,0], label=labels[i], color=cdict[labels[i]])#'blue' if labels[i] == 1 else 'green')
    plt.ylim([0,4000])
    plt.legend()
# %%
summed_clusters = {}
for g in np.unique(labels):
    summed_clusters[g] = np.zeros(dataset_small.shape[1])
    for l in np.where(labels == g):
        for i in range(len(l)):
            temp = dataset_small[l[i],:,0]
            # print(np.shape(temp))
            summed_clusters[g] += temp

# %%
dishwasher_index_list = []
for i in summed_clusters.keys():
    dtw_score_dishwasher = dtw(summed_clusters[i], dishwasher_omp)
    print('DTW Dishwasher Score ' + str(i) + ' = ' +str(dtw_score_dishwasher))
    if dtw_score_dishwasher < 2000:
        dishwasher_index_list.append(i)
dishwasher_index_list
# %%
for i in summed_clusters.keys():
    if i == dishwasher_index_list:
        plt.plot(summed_clusters[i], color='green', 
        label='Dishwasher')
    else:
        plt.plot(summed_clusters[i], color='blue', 
        label='Leftover')
    plt.ylim([0,4000])
    plt.legend()
# %%
final_summed_clusters = np.zeros(summed_clusters[0].shape[0])
for i in summed_clusters.keys():
    if i == dishwasher_index_list:
        continue
    else:
        final_summed_clusters += summed_clusters[i]
# %%
washer0_start = np.where(summed_washer_clusters[0])[0][0]
washer1_start = np.where(summed_washer_clusters[1])[0][0]

if washer0_start < washer1_start:
    washer_dtw = summed_washer_clusters[0]
elif washer0_start > washer1_start:
    washer_dtw = summed_washer_clusters[1]

washerdryer_index_list = []
for i in summed_clusters.keys():
    dtw_score_washerdryer = dtw(summed_clusters[i], washer_dtw)
    print('Washer Dryer DTW Score ' + str(i) + ' = ' +str(dtw_score_washerdryer))
    if dtw_score_washerdryer < 5000:
        washerdryer_index_list.append(i)
washerdryer_index_list
# print('I think if under 2000, we can classify it as washer dryer')
# %%
# how to also include the smaller parts of the washer dryer signature when finding the initial starting point
fig, ax = plt.subplots()

ax.plot(final_summed_clusters, color='green', label='Washer Dryer')
plt.ylim([0,4000])
plt.legend()
plt.show()
# %%
########## FINAL PLOT ############
fig, ax = plt.subplots()
ax.plot(agg,'--')

for j in kettles_index_list:
    ax.plot(result_final.Kcoef[:,j], color='purple', 
    label='Kettle' if j == kettles_index_list[0] else "")
for i in microwave_index_list:
    ax.plot(result_final.Kcoef[:,i], color='red', 
    label='Microwave' if i == microwave_index_list[0] else "")
for k in fridgefreezer_index_list:
    ax.plot(result_final.Kcoef[:,k], color='green', 
    label='Fridgefreezer' if k == fridgefreezer_index_list[0] else "")
for z in summed_clusters.keys():
    if z == dishwasher_index_list:
        ax.plot(summed_clusters[z], color='black', 
        label='Dishwasher')
    elif z == washerdryer_index_list or z != dishwasher_index_list:
        ax.plot(summed_clusters[z], color='darkorange', label='Washer Dryer' if z == washerdryer_index_list[0] else "")

plt.ylim([0, 4000])
plt.legend()
plt.show()
# %%

for j in kettles_index_list:
    location = np.where(result_final.Kcoef[:,j] != 0)
    loc = location[0][0]
    for i in range(len(agg_actual['kettle'])):
        if (agg_actual['kettle'][i] - 100) < loc < (agg_actual['kettle'][i] + 100):
            print('Kettle True at ' + str(loc))

for j in microwave_index_list:
    location = np.where(result_final.Kcoef[:,j] != 0)
    loc = location[0][0]
    for i in range(len(agg_actual['microwave'])):
        if (agg_actual['microwave'][i] - 100) < loc < (agg_actual['microwave'][i] + 100):
            print('Microwave True at ' + str(loc))

for j in fridgefreezer_index_list:
    location = np.where(result_final.Kcoef[:,j] != 0)
    loc = location[0][0]
    for i in range(len(agg_actual['fridgefreezer'])):
        if (agg_actual['fridgefreezer'][i] - 100) < loc < (agg_actual['fridgefreezer'][i] + 100):
            print('Fridge Freezer True at ' + str(loc))

for z in dishwasher_index_list:
    location = np.where(summed_clusters[z] != 0)
    loc = location[0][0]
    if (agg_actual['dishwasher'][0] - 100) < (loc - 198) < (agg_actual['dishwasher'][0] + 100):
        print('Dishwasher True at ' + str(loc))
    else:
        print(loc)

for k in washerdryer_index_list:
    location = np.where(summed_clusters[k] != 0)
    loc = location[0][0]
    if (agg_actual['washerdryer'][0] - 100) < loc < (agg_actual['washerdryer'][0] + 100):
        print('Washer Dryer True at ' + str(loc))
    else:
        print(loc)

agg_actual
# %%
import numpy as np
from ksvd import ApproximateKSVD


# X ~ gamma.dot(dictionary)
X = np.random.randn(1000, 20)
aksvd = ApproximateKSVD(n_components=20)
dictionary = aksvd.fit(smooth_result.Kcoef).components_
gamma = aksvd.transform(smooth_result.Kcoef)
# %%
ksvd = gamma.dot(dictionary)
newagg = np.zeros(ksvd.shape[0])
for i in range(ksvd.shape[1]):
    newagg += ksvd[:,i]
# %%
import sklearn

import numpy as np
from sklearn.datasets import make_sparse_coded_signal
from sklearn.decomposition import DictionaryLearning
# %%
X, dictionary, code = make_sparse_coded_signal(n_samples=100, n_components=15, n_features=20, n_nonzero_coefs=10, random_state=42)
dict_learner = DictionaryLearning(
    n_components=16, transform_algorithm='lasso_lars')#, random_state=42,)
X_transformed = dict_learner.fit_transform(result_final.Kcoef)
# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import SparseCoder
from sklearn.utils.fixes import np_version, parse_version
# %%

def ricker_function(resolution, center, width):
    """Discrete sub-sampled Ricker (Mexican hat) wavelet"""
    x = np.linspace(0, resolution - 1, resolution)
    x = ((2 / (np.sqrt(3 * width) * np.pi ** .25))
         * (1 - (x - center) ** 2 / width ** 2)
         * np.exp(-(x - center) ** 2 / (2 * width ** 2)))
    return x


def ricker_matrix(width, resolution, n_components):
    """Dictionary of Ricker (Mexican hat) wavelets"""
    centers = np.linspace(0, resolution - 1, n_components)
    D = np.empty((n_components, resolution))
    for i, center in enumerate(centers):
        D[i] = ricker_function(resolution, center, width)
    D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
    return D


resolution = 2000
subsampling = 3  # subsampling factor
width = 100
n_components = resolution // subsampling

# Compute a wavelet dictionary
D_fixed = ricker_matrix(width=width, resolution=resolution,
                        n_components=n_components)
D_multi = np.r_[tuple(ricker_matrix(width=w, resolution=resolution,
                      n_components=n_components // 5)
                for w in (10, 50, 100, 500, 1000))]
# D_fixed = dictionary.T
# Generate a signal
# y = np.linspace(0, resolution - 1, resolution)
# first_quarter = y < resolution / 4
# y[first_quarter] = 3.
# y[np.logical_not(first_quarter)] = -1.
# %%
y = agg

# List the different sparse coding methods in the following format:
# (title, transform_algorithm, transform_alpha,
#  transform_n_nozero_coefs, color)
estimators = [('OMP', 'omp', None, 15, 'navy'),
              ('Lasso', 'lasso_lars', 2, None, 'turquoise'), ]
lw = 2
# Avoid FutureWarning about default value change when numpy >= 1.14
lstsq_rcond = None if np_version >= parse_version('1.14') else -1

plt.figure(figsize=(13, 6))
for subplot, (D, title) in enumerate(zip((D_fixed, D_multi),
                                         ('fixed width', 'multiple widths'))):
    plt.subplot(1, 2, subplot + 1)
    plt.title('Sparse coding against %s dictionary' % title)
    plt.plot(y, lw=lw, linestyle='--', label='Original signal')
    # Do a wavelet approximation
    for title, algo, alpha, n_nonzero, color in estimators:
        coder = SparseCoder(dictionary=D, transform_n_nonzero_coefs=n_nonzero,
                            transform_alpha=alpha, transform_algorithm=algo)
        x = coder.transform(y.reshape(1, -1))
        density = len(np.flatnonzero(x))
        x = np.ravel(np.dot(x, D))
        squared_error = np.sum((y - x) ** 2)
        plt.plot(x, color=color, lw=lw,
                 label='%s: %s nonzero coefs,\n%.2f error'
                 % (title, density, squared_error))
        print('Done 1')

    # Soft thresholding debiasing
    coder = SparseCoder(dictionary=D, transform_algorithm='threshold',
                        transform_alpha=20)
    x = coder.transform(y.reshape(1, -1))
    _, idx = np.where(x != 0)
    x[0, idx], _, _, _ = np.linalg.lstsq(D[idx, :].T, y, rcond=lstsq_rcond)
    x = np.ravel(np.dot(x, D))
    squared_error = np.sum((y - x) ** 2)
    plt.plot(x, color='darkorange', lw=lw,
             label='Thresholding w/ debiasing:\n%d nonzero coefs, %.2f error'
             % (len(idx), squared_error))
    plt.axis('tight')
    plt.legend(shadow=False, loc='best')
plt.subplots_adjust(.04, .07, .97, .90, .09, .2)
plt.show()
# %%
estimators = [('OMP', 'omp', None, 15, 'navy'),
              ('Lasso', 'lasso_lars', 2, None, 'turquoise'), ]

lw = 2
# Avoid FutureWarning about default value change when numpy >= 1.14
lstsq_rcond = None if np_version >= parse_version('1.14') else -1

# plt.figure(figsize=(13, 6))
# for subplot, (D, title) in enumerate(zip((D_fixed, D_multi),
#                                          ('fixed width', 'multiple widths'))):

#     plt.subplot(1, 2, subplot + 1)
#     plt.title('Sparse coding against %s dictionary' % title)
#     plt.plot(y, lw=lw, linestyle='--', label='Original signal')
    # Do a wavelet approximation
for title, algo, alpha, n_nonzero, color in estimators:
    coder = SparseCoder(dictionary=dictionary.T, transform_n_nonzero_coefs=n_nonzero,
                        transform_alpha=alpha, transform_algorithm=algo)
    x = coder.transform(y.reshape(1, -1))
    density = len(np.flatnonzero(x))
    x = np.ravel(np.dot(x, dictionary.T))
    squared_error = np.sum((y - x) ** 2)
    plt.plot(x, color=color, lw=lw,
                label='%s: %s nonzero coefs,\n%.2f error'
                % (title, density, squared_error))
    print('Done 1')
# %%
agg = open_pickle('/Users/kayvon/Desktop/agg3.pkl')

# %%
lw = 2
lstsq_rcond = None if np_version >= parse_version('1.14') else -1

    # Soft thresholding debiasing
coder = SparseCoder(dictionary=dictionary2.T, transform_algorithm='lasso_lars',
                    transform_alpha=200)
x = coder.transform(agg.reshape(1, -1))
_, idx = np.where(x != 0)
x[0, idx], _, _, _ = np.linalg.lstsq(dictionary2.T[idx, :].T, agg, rcond=lstsq_rcond)
x = np.ravel(np.dot(x, dictionary2.T))
squared_error = np.sum((agg - x) ** 2)
# %%
plt.plot(x, color='darkorange', lw=lw,
            label='Thresholding w/ debiasing:\n%d nonzero coefs, %.2f error'
            % (len(idx), squared_error))
plt.axis('tight')
plt.legend(shadow=False, loc='best')
plt.subplots_adjust(.04, .07, .97, .90, .09, .2)
plt.show()
# %%
dictionary2 = np.zeros((2000,100500))
cnt = 0
for i in np.arange(0,402000,4):
    dictionary2[:,cnt] = dictionary[:,i]
    cnt += 1

# %%
dictionary = open_pickle('/Users/kayvon/Desktop/dictionary2000.pkl')
dictionary_del = dictionary

for i in range(dictionary.shape[1]):
    if max(dictionary[:,i]) == 0:
        dictionary_del = np.delete(dictionary, i, axis=1)
# %%
