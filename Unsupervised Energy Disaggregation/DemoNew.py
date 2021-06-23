#! usr/bin/env python3
# %%
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math
import re
import time
import matplotlib.cm as cm
import dill
import pickle
import scipy.optimize as sci
import seaborn as sns
sns.set()
import statistics
from tslearn.metrics import dtw, soft_dtw
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import linalg


import dill
def save_pickle(obj,file_name):
    with open(file_name, 'wb') as handle:
        dill.dump(obj, handle, protocol=dill.HIGHEST_PROTOCOL)
    return file_name + ' is saved'

def open_pickle(file_name):
    with open(file_name, 'rb') as handle:
        obj = dill.load(handle)
    return obj

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

def Heaviside(x, a):
    """Compute a Heaviside function."""
    y = (np.sign(x - a) + 1) / 2
    y[y == 0.5] = 0
    return y

def Boxcar(l, w, t):
    """Compute a boxcar function.

    Arguments:
        l -- constant in the interval (non-zero)
        w -- width of the boxcar, i.e. the internval equal to constant 'l'
        t -- sequence of the time horizon

    Returns:
        Vector with the set parameter for the boxcar
    """
    a = l - w / 2
    b = l + w / 2

    if a == b:
        H = np.zeros(shape=(len(t),), dtype=float)
        H[a] = 1.0

    else:
        H = Heaviside(t, a) - Heaviside(t, b)

    return(1 / np.sqrt(w) * H)

def gen_dict2(tslength, infos=False, boxwidth=120):
    """Generate a dictionary.

    Arguments:
        tslength: time series length
        infos: if 'True' return the information about the boxcar boxwidth
    """

    # Time vector
    x = np.linspace(1, tslength, tslength)

    # Dictionary matrix
    X1 = np.eye(tslength)
    X2 = np.zeros((tslength, tslength), dtype=float)
    X = np.concatenate((X1, X2), axis=1)

    if tslength < boxwidth:
        boxwidth = tslength

    #boxcarinfos = np.zeros((tslength * boxwidth, 2), dtype=float)

    for j in range(1, boxwidth):
        for i in range(1, tslength):
            X2[:, i] = Boxcar(i, j, x)
            # boxcarinfos[j,0] = j
            # boxcarinfos[i,1] = i
        X = np.concatenate((X, X2), axis=1)

    if not infos:
        return X

    else:
        return X, boxcarinfos

def cholesky_omp(D, x, m, eps=None):
    if eps == None:
        stopping_condition = lambda: it == m  # len(idx) == m
    else:
        stopping_condition = lambda: np.inner(residual, residual) <= eps

    alpha = np.dot(x, D)
    
    #first step:        
    it = 1
    lam = np.abs(np.dot(x, D)).argmax()
    idx = [lam]
    L = np.ones((1,1))
    gamma = linalg.lstsq(D[:, idx], x)[0]
    residual = x - np.dot(D[:, idx], gamma)
    
    while not stopping_condition():
        lam = np.abs(np.dot(residual, D)).argmax()
        w = linalg.solve_triangular(L, np.dot(D[:, idx].T, D[:, lam]),
                                    lower=True, unit_diagonal=True)
        # should the diagonal be unit in theory? It crashes without it
        L = np.r_[np.c_[L, np.zeros(len(L))],
                  np.atleast_2d(np.append(w, np.sqrt(1 - np.dot(w.T, w))))]
        idx.append(lam)
        it += 1
        #gamma = linalg.solve(np.dot(L, L.T), alpha[idx], sym_pos=True)
        # what am I, stupid??
        Ltc = linalg.solve_triangular(L, alpha[idx], lower=True)
        gamma = linalg.solve_triangular(L, Ltc, trans=1, lower=True)
        residual = x - np.dot(D[:, idx], gamma)

    return gamma, idx
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

half_washer = np.zeros((result['washerdryer'].Kcoef.shape[1],result['washerdryer'].Kcoef.shape[0],1))
for i in range(result['washerdryer'].Kcoef.shape[1]):
    half_washer[i,:,0] = result['washerdryer'].Kcoef[:,i]
# half_washer += result['washerdryer'].Kcoef

dba_km = TimeSeriesKMeans(n_clusters=2,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10)
y_pred = dba_km.fit_predict(half_washer)

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
        for j in range(1):
            num = np.random.randint(2000-len(noisy[i]))
            agg[num:num+noisy[i].shape[0]] += noisy[i]
            print('Fridge Freezer placement = ' + str(num))
            agg_actual[i][j] = num
    elif i == 'kettle':
        for k in range(1):
            num = np.random.randint(2000-len(noisy[i]))
            agg[num:num+noisy[i].shape[0]] += noisy[i]
            print('Kettle placement = ' + str(num))
            agg_actual[i][k] = num
    elif i == 'microwave':
        for z in range(1):
            num = np.random.randint(2000-len(noisy[i]))
            agg[num:num+noisy[i].shape[0]] += noisy[i]
            print('Microwave placement = ' + str(num))
            agg_actual[i][z] = num


########### PLOT AGGREGATE ############
plt.plot(agg)
plt.ylim([0,max(agg)])
#### TABLE FOR APPLIANCE PLACEMENT ####
agg_actual
# %%
start_time = time.time()
df1 = pd.DataFrame()
result_final = OMP(x=agg, Dictionary=dictionary, dataframe=df1, maxiter=1000, tol=0.002, S=30, threshold_min_power=0)
print('--- %s seconds ---' % (time.time() - start_time))
# %%
agg_df = open_pickle('/Users/kayvon/Desktop/agg_df2.pkl')
agg = agg_df['agg'][14452*16+10000:14452*17+10000]
# %%
ind = np.where(agg <= min(agg)+50)[0]
index = [0]
for i in np.arange(1,len(ind)):
    if ind[i] - ind[i-1] >= 50:
        index.append(ind[i])
index = index[0:len(index)-1]
index.append(len(agg))
# %%
startio = time.time()
# %%
############### MAIN ITERATION ###############
m = 5 # Number of boxes to stop at
eps = 0.55
selected_boxes = {}
total_agg_approx = []
tot_start = time.time()
for i in np.arange(1,len(index)):
    start_time = time.time()
    print('Length of Dictionary = ' + str(index[i] - index[i-1]))
    dictionary = gen_dict2(tslength=(index[i]-index[i-1]), infos=False, boxwidth=200)
    print('--- %s seconds ---' % (time.time() - start_time))
    start_time1 = time.time()
    gamma, idx = cholesky_omp(D=dictionary,x=agg[index[i-1]:index[i]],m=m)
    print('Cholesky OPM Approximated')
    print('--- %s seconds ---' % (time.time() - start_time1))
    print(' ')
    agg_approximated = np.zeros(len(agg[index[i-1]:index[i]]))
    length = len(selected_boxes)
    for j in range(len(idx)):
        temp = np.dot(dictionary[:,idx[j]],gamma[j])
        agg_approximated += temp
        selected_boxes[length + j] = temp
    total_agg_approx.extend(agg_approximated)
print('--- %s seconds total ---' % (time.time() - tot_start))
# %%
final_boxes = {}
final_boxes[0] = []
cnt = 0
for i in np.arange(1,len(selected_boxes)+1):
    if i != (len(selected_boxes)):
        if len(selected_boxes[i]) == len(selected_boxes[i-1]):
            final_boxes[cnt].append(selected_boxes[i-1])
        else:
            final_boxes[cnt].append(selected_boxes[i-1])
            cnt += 1
            final_boxes[cnt] = []
    else:
        final_boxes[cnt].append(selected_boxes[i-1])
#%%
import copy
final_boxes1 = copy.deepcopy(final_boxes)
# %%
# save_pickle(final_boxes, '/Users/kayvon/Desktop/final_boxes_test2.pkl')
# final_boxes = open_pickle('/Users/kayvon/Desktop/final_boxes_test2.pkl')
# final_boxes1 = open_pickle('/Users/kayvon/Desktop/final_boxes_test2.pkl')
# %%

ts_length = 0
for i in range(len(final_boxes1)):
    ts_length1 = ts_length
    ts_length += len(final_boxes1[i][0])
    for j in range(len(final_boxes1[i])):
        final_boxes1[i][j] = np.pad(final_boxes1[i][j], (ts_length1,len(agg)-ts_length), 'constant', constant_values=(0))
        # print(ts_length)

df_final_boxes_full_length = pd.DataFrame()
cnt = 0
for i in range(len(final_boxes1)):
    for j in range(len(final_boxes1[i])):
        df_final_boxes_full_length[cnt] = final_boxes1[i][j]
        cnt += 1
# %%
lens = []
for i in range(len(final_boxes)):
    lens.append(len(final_boxes[i][0]))
(max(lens))

for i in range(len(final_boxes)):
    for j in range(len(final_boxes[i])):
        final_boxes[i][j] = np.pad(final_boxes[i][j],(0,(max(lens)-len(final_boxes[i][j]))),'constant', constant_values=(0))

df_final_boxes = pd.DataFrame()
cnt = 0
for i in range(len(final_boxes)):
    for j in range(len(final_boxes[i])):
        df_final_boxes[cnt] = final_boxes[i][j]
        cnt += 1
# %%
# final_boxes = {}
# cnt = 0
# for i in range(len(selected_boxes.keys())-1):
#     final_boxes[cnt] = []
#     if len(selected_boxes[i]) == len(selected_boxes[i+1]):
#         final_boxes[cnt].append(selected_boxes[i])
#     else:
#         cnt += 1
# %%
agg_boxes = []
for i in selected_boxes.keys():
    agg_boxes.extend(selected_boxes[i])
# %%
start_time = time.time()
dictionary = gen_dict2(len(agg), infos=False, boxwidth=200)
print('--- %s seconds ---' % (time.time() - start_time))
# %%
# https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf
# https://gist.github.com/vene/996771
start_time = time.time()
gamma, idx = cholesky_omp(D=dictionary,x=agg,m=30)
print('--- %s seconds ---' % (time.time() - start_time))

agg_approximated = np.zeros(len(agg))
selected_boxes = pd.DataFrame()

for i in range(len(idx)):
    temp = np.dot(dictionary[:,idx[i]],gamma[i])
    agg_approximated += temp
    selected_boxes[i] = temp

plt.plot(selected_boxes);
# %%
# %%
start_time = time.time()
sparse_codes = dictlearn.omp_cholesky(agg,dictionary,n_nonzero=30,tol=0.055)
sparse_approx = dictionary.dot(sparse_codes)
print('--- %s seconds ---' % (time.time() - start_time))
#%%
selected_boxes = pd.DataFrame()
for i in np.where(sparse_codes[:] != 0)[0]:
    selected_boxes[i] = np.dot(sparse_codes[i],dictionary[:,i])

plt.plot(selected_boxes);

# %%
mean_squared_error(agg,total_agg_approx)
# %%
mean_squared_error(agg,result_final.RecSignal)
# %%
selected_boxes = df_final_boxes
# %%
kettles_index_list = []
for i in range(selected_boxes.shape[1]):
    dtw_score_kettle = dtw(selected_boxes.iloc[:,i], kettle_omp)
    # print('Kettle DTW Score ' + str(i) + '= ' +str(dtw_score_kettle))
    if dtw_score_kettle < 2000:
        ind = np.where(selected_boxes.iloc[:,i] != 0)
        start = ind[0][0] - 2
        end = ind[0][-1] + 2
        if (len(kettle_omp) - (0.5*len(kettle_omp))) <= (end - start) <= (len(kettle_omp) + (0.5*len(kettle_omp))):
            dtw_kettle = dtw(selected_boxes.iloc[:,i], kettle_omp)
            print('Kettle DTW Score ' + str(i) + '= ' +str(dtw_kettle))
            if dtw_kettle < 460:
                kettles_index_list.append(i)
        else:
            continue
        # kettles_index_list.append(i)
kettles_index_list
# %%
microwave_index_list = []
for i in range(selected_boxes.shape[1]):
    dtw_score_microwave = dtw(selected_boxes.iloc[:,i], microwave_omp)
    # print('Kettle DTW Score ' + str(i) + '= ' +str(dtw_score_kettle))
    if dtw_score_microwave < 2000:
        ind = np.where(selected_boxes.iloc[:,i] != 0)
        start = ind[0][0]
        end = ind[0][-1]
        if ((len(np.where(microwave_omp != 0)[0]) - (0.7*len(np.where(microwave_omp != 0)[0])))
             <= (end - start) <=
            (len(np.where(microwave_omp != 0)[0]) + (0.7*len(np.where(microwave_omp != 0)[0])))):
            print(end - start)
            dtw_microwave = dtw(selected_boxes.iloc[:,i], microwave_omp)
            print('Microwave DTW Score ' + str(i) + '= ' +str(dtw_microwave))
            if dtw_microwave < 500:
                microwave_index_list.append(i)
        else:
            continue

microwave_index_list
# %%
fridgefreezer_index_list = []
for i in range(selected_boxes.shape[1]):
    dtw_score_fridgefreezer = dtw(selected_boxes.iloc[:,i], fridgefreezer_omp)
    # print('Kettle DTW Score ' + str(i) + '= ' +str(dtw_score_kettle))
    if dtw_score_fridgefreezer < 100:
        ind = np.where(selected_boxes.iloc[:,i] != 0)
        start = ind[0][0]
        end = ind[0][-1]
        if ((len(np.where(fridgefreezer_omp != 0)[0]) - (0.7*len(np.where(fridgefreezer_omp != 0)[0]))) 
            <= (end - start) <= 
            (len(np.where(fridgefreezer_omp != 0)[0]) + (0.7*len(np.where(fridgefreezer_omp != 0)[0])))):
            dtw_fridgefreezer = dtw(selected_boxes.iloc[:,i], fridgefreezer_omp)
            print('Fridge Freezer DTW Score ' + str(i) + '= ' +str(dtw_fridgefreezer))
            fridgefreezer_index_list.append(i)
        else:
            continue

fridgefreezer_index_list
# %%
fig, ax = plt.subplots()

for j in kettles_index_list:
    ax.plot(selected_boxes.iloc[:,j], color='blue', 
    label='Kettle' if j == kettles_index_list[0] else "")
for i in microwave_index_list:
    ax.plot(selected_boxes.iloc[:,i], color='red', 
    label='Microwave' if i == microwave_index_list[0] else "")
for k in fridgefreezer_index_list:
    ax.plot(selected_boxes.iloc[:,k], color='green', 
    label='Fridgefreezer' if k == fridgefreezer_index_list[0] else "")

plt.ylim([0, max(agg)])
plt.legend()
plt.show()
# %%
indexes = np.arange(0,selected_boxes.shape[1]).tolist()
type1_indexes = []
for i in range(len(kettles_index_list)):
    type1_indexes.append(kettles_index_list[i])
for j in range(len(microwave_index_list)):
    type1_indexes.append(microwave_index_list[j])
for k in range(len(fridgefreezer_index_list)):
    type1_indexes.append(fridgefreezer_index_list[k])


for element in type1_indexes:
    if element in indexes:
        indexes.remove(element)
# %%
# fig, ax = plt.subplots()
# for i in indexes:
#     ax.plot(selected_boxes.iloc[:,i])

# plt.ylim([0,max(agg)])
# plt.show()

# leftover = np.zeros(len(agg))
# for i in indexes:
#     temp = selected_boxes.iloc[:,i]
#     leftover += temp

# fig, ax = plt.subplots()

# ax.plot(leftover, label='Leftover')
# plt.ylim([0,max(agg)])
# plt.legend()
# plt.show()

# %%
start_time = time.time()
dataset_small = np.zeros((selected_boxes.shape[1],selected_boxes.shape[0],1))

for i in indexes:
    dataset_small[i,:,0] = selected_boxes.iloc[:,i]

df_dataset_small = pd.DataFrame(dataset_small[:,:,0])
df_dataset_small = df_dataset_small.drop(type1_indexes, axis=0)
df_dataset_small = df_dataset_small.reset_index(drop=False)

dataset_small = np.delete(dataset_small, type1_indexes, axis=0)


# %%
dba_km1 = TimeSeriesKMeans(n_clusters=3,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10, n_jobs=-1)
y_pred = dba_km1.fit_predict(dataset_small)

labels = dba_km1.labels_

print('--- %s seconds ---' % (time.time() - start_time))
# %%
cdict = {0: 'red', 1: 'blue', 2: 'green'}
for g in np.unique(labels):
    for i in range(len(dataset_small)):
        plt.plot(dataset_small[i,:,0], label=labels[i] if labels[i] == g else "", color=cdict[labels[i]])
        plt.ylim([-2000,max(agg)])
        plt.legend()
# %%
# right after labeling go back with indexes and find actual positions of 
# boxes for dishwasher index list so that the boxes arent on top of each other

clusters = {}
for g in np.unique(labels):
    clusters[g] = []
    clusters[g].append(np.where(labels==g)[0])
# %%
cluster0 = []
cluster1 = []
cluster2 = []
for i in clusters.keys():
    if i == 0:
        for j in range(len(clusters[i][0])):
            position = df_dataset_small.iloc[clusters[i][0][j],0]
            cluster0.append(position)
    if i == 1:
        for j in range(len(clusters[i][0])):
            position = df_dataset_small.iloc[clusters[i][0][j],0]
            cluster1.append(position)
    if i == 2:
        for j in range(len(clusters[i][0])):
            position = df_dataset_small.iloc[clusters[i][0][j],0]
            cluster2.append(position)        
# %%
summed_clusters = {}
for g in np.unique(labels):
    if g == 0:
        summed_clusters[g] = np.zeros(df_final_boxes_full_length.shape[0])
        for i in range(len(cluster0)):
            temp = df_final_boxes_full_length.iloc[:,cluster0[i]]
            if i == 0:
                summed_clusters[g] += temp
            else:
                if np.where(abs(temp) > 0)[0][-1] - np.where(abs(summed_clusters[g]) > 0)[0][-1] < 1000:
                    summed_clusters[g] += temp
                else:
                    continue
        if len(np.where(summed_clusters[g] > 0)[0]) < 100:
            summed_clusters[g] = np.zeros(df_final_boxes_full_length.shape[0])
            for i in range(len(cluster0)):
                temp = df_final_boxes_full_length.iloc[:,cluster0[i]]
                summed_clusters[g] += temp

    if g == 1:
        summed_clusters[g] = np.zeros(df_final_boxes_full_length.shape[0])
        for i in range(len(cluster1)):
            temp = df_final_boxes_full_length.iloc[:,cluster1[i]]
            if i == 0:
                summed_clusters[g] += temp
            else:
                # print(np.where(abs(temp)>0)[0])
                # print(np.where(summed_clusters[g]>0)[0])
                # print(i)
                if np.where(abs(temp) > 0)[0][-1] - np.where(abs(summed_clusters[g]) > 0)[0][-1] < 1000:
                    summed_clusters[g] += temp
                else:
                    continue
        if len(np.where(summed_clusters[g] > 0)[0]) < 100:
            summed_clusters[g] = np.zeros(df_final_boxes_full_length.shape[0])
            for i in np.arange(1,len(cluster1)):
                temp = df_final_boxes_full_length.iloc[:,cluster1[i]]
                summed_clusters[g] += temp

    if g == 2:
        summed_clusters[g] = np.zeros(df_final_boxes_full_length.shape[0])
        for i in range(len(cluster2)):
            temp = df_final_boxes_full_length.iloc[:,cluster2[i]]
            if i == 0:
                summed_clusters[g] += temp
            else:
                if np.where(abs(temp) > 0)[0][-1] - np.where(abs(summed_clusters[g]) > 0)[0][-1] < 1000:
                    summed_clusters[g] += temp
                else:
                    continue
        if len(np.where(summed_clusters[g] > 0)[0]) < 100:
            summed_clusters[g] = np.zeros(df_final_boxes_full_length.shape[0])
            for i in range(len(cluster2)):
                temp = df_final_boxes_full_length.iloc[:,cluster2[i]]
                summed_clusters[g] += temp

# %%
# summed_clusters = {}
# for g in np.unique(labels):
#     summed_clusters[g] = np.zeros(dataset_small.shape[1])
#     for l in np.where(labels == g):
#         for i in range(len(l)):
#             temp = dataset_small[l[i],:,0]
#             # print(np.shape(temp))
#             summed_clusters[g] += temp

# %%
dishwasher_index_list = []
for i in summed_clusters.keys():
    dtw_score_dishwasher = dtw(summed_clusters[i], dishwasher_omp)
    print('DTW Dishwasher Score ' + str(i) + ' = ' +str(dtw_score_dishwasher))
    if dtw_score_dishwasher < 2500:
        dishwasher_index_list.append(i)

print(dishwasher_index_list)

if len(dishwasher_index_list) != 0:
    dishwasher_index_list_final = []
    if dishwasher_index_list[0] == 0:
        dishwasher_index_list_final.extend(cluster0)
    if dishwasher_index_list[0] == 1:
        dishwasher_index_list_final.extend(cluster1)
    if dishwasher_index_list[0] == 2:
        dishwasher_index_list_final.extend(cluster2)

    for i in summed_clusters.keys():
        if i == dishwasher_index_list:
            plt.plot(summed_clusters[i], color='green', 
            label='Dishwasher')
        else:
            plt.plot(summed_clusters[i], color='blue', 
            label='Leftover')
        plt.ylim([0,max(agg)])
        plt.legend()
# %%
# final_summed_clusters = np.zeros(summed_clusters[0].shape[0])
# for i in summed_clusters.keys():
#     if i == dishwasher_index_list:
#         continue
#     else:
#         final_summed_clusters += summed_clusters[i]

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
print(washerdryer_index_list)

if len(washerdryer_index_list) != 0:
    washerdryer_index_list_final = []
    if washerdryer_index_list[0] == 0:
        washerdryer_index_list_final.extend(cluster0)
    if washerdryer_index_list[0] == 1:
        washerdryer_index_list_final.extend(cluster1)
    if washerdryer_index_list[0] == 2:
        washerdryer_index_list_final.extend(cluster2)
# %%
print('--- %s seconds ---' % (time.time() - startio))
# %%
for j in [0,-1]:
    for i in dishwasher_index_list_final:
        print(np.where(df_final_boxes_full_length.iloc[:,i]>=100)[0][0])
print(np.where(agg_df['dishwasher'][14452*5+10000:14452*6+10000]>=100)[0][0])
# %%
fig, ax = plt.subplots()

ax.plot(summed_clusters[washerdryer_index_list[0]], color='green', label='Washer Dryer')
plt.ylim([0,max(agg)])
plt.legend()
plt.show()
# %%
# %%
kettles = np.zeros(df_final_boxes_full_length.shape[0])
for i in kettles_index_list:
    kettles += (df_final_boxes_full_length.iloc[:,i])
# %%
microwaves = np.zeros(df_final_boxes_full_length.shape[0])
for i in microwave_index_list:
    microwaves += (df_final_boxes_full_length.iloc[:,i])

# %%
fridges = np.zeros(df_final_boxes_full_length.shape[0])
for i in fridgefreezer_index_list:
    fridges += (df_final_boxes_full_length.iloc[:,i])

# %%

# %%
mean_absolute_error(agg_df['washerdryer'][14452*16+10000:14452*17+10000],np.zeros(14452))
# %%
mean_squared_error(agg_df['washerdryer'][14452*16+10000:14452*17+10000],np.zeros(14452), squared=False)
# %%
ec_num = 0
ec_den = 0
for i in range(len(summed_clusters[1])):
    ec_num += abs(np.zeros(14452)[i] - agg_df['washerdryer'][14452*16+10000+i])
    ec_den += agg_df['washerdryer'][14452*16+10000+i]

EC = 1 - (ec_num/(2*ec_den))

EC
# %%
np.zeros(14452)
# %%
########## FINAL PLOT ############
fig, ax = plt.subplots()
ax.plot(agg,'--')

for j in kettles_index_list:
    ax.plot(selected_boxes.iloc[:,j], color='purple', 
    label='Kettle' if j == kettles_index_list[0] else "")
for i in microwave_index_list:
    ax.plot(selected_boxes.iloc[:,i], color='red', 
    label='Microwave' if i == microwave_index_list[0] else "")
for k in fridgefreezer_index_list:
    ax.plot(selected_boxes.iloc[:,k], color='green', 
    label='Fridgefreezer' if k == fridgefreezer_index_list[0] else "")
for z in summed_clusters.keys():
    if z == dishwasher_index_list:
        ax.plot(summed_clusters[z], color='black', 
        label='Dishwasher')
    elif z == washerdryer_index_list or z != dishwasher_index_list:
        ax.plot(summed_clusters[z], color='darkorange', label='Washer Dryer' if z == washerdryer_index_list[0] else "")

plt.ylim([0, max(agg)])
plt.legend()
plt.show()
# %%
############# CHECK IF CORRECT ############
for j in kettles_index_list:
    location = np.where(selected_boxes.iloc[:,j] != 0)
    loc = location[0][0]
    for i in range(len(agg_actual['kettle'])):
        if (agg_actual['kettle'][i] - 100) < loc < (agg_actual['kettle'][i] + 100):
            print('Kettle True at ' + str(loc))

for j in microwave_index_list:
    location = np.where(selected_boxes.iloc[:,j] != 0)
    loc = location[0][0]
    for i in range(len(agg_actual['microwave'])):
        if (agg_actual['microwave'][i] - 100) < loc < (agg_actual['microwave'][i] + 100):
            print('Microwave True at ' + str(loc))

for j in fridgefreezer_index_list:
    location = np.where(selected_boxes.iloc[:,j] != 0)
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

# %%
