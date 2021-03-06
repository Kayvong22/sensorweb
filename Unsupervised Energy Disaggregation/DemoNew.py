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
    ''' Cholesky OMP Approximation
    Arguments:
        D: Dictionary
        x: signal
        m: # of boxcars for stopping criteria
        eps: residual for stopping criteria
            Use m or eps; if eps is not None, m won't be used
    '''
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
########## SYNTHETIC NOISY AGGREGATE ###########
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
agg = agg_df['agg'][14452*14+10000:14452*15+10000]
# %%
ind = np.where(agg <= min(agg)+50)[0]
index = [0]
for i in np.arange(1,len(ind)):
    if ind[i] - ind[i-1] >= 10:
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
cdict = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange'}
for g in np.unique(labels):
    for i in range(len(dataset_small)):
        plt.plot(dataset_small[i,:,0], label=labels[i] if labels[i] == g else "", color=cdict[labels[i]])
        plt.ylim([-2000,max(agg)])
        plt.legend()
# %%
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











## You don't need the rest of this
# %%

fig, axs = plt.subplots(3, 1)
fig.set_figheight(10)
fig.set_figwidth(15)

axs[0].set_title('Actual Aggregate Signal')
axs[0].set_ylim([-150,3800])
axs[0].plot(np.arange(0,14452),agg, 'r')
axs[1].set_title('Cholesky OMP Selected Boxcars')
for i in range(df_final_boxes_full_length.shape[1]):
    axs[1].plot(df_final_boxes_full_length.iloc[:,i]);
axs[2].set_ylim([-150,3800])
axs[2].set_title('Cholesky OMP Approximation')
axs[2].plot(total_agg_approx)

for ax in axs.flat:
    ax.set(xlabel='Samples (1/6 Hz each index)', ylabel='Power (Watts)')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

# %%
kettle_omp1 = np.pad(kettle_omp, (0,len(dishwasher_omp) - len(kettle_omp)),mode='constant', constant_values=(0))
fridgefreezer_omp1 = np.pad(fridgefreezer_omp, (0, len(dishwasher_omp) - len(fridgefreezer_omp)), mode='constant', constant_values=(0))
microwave_omp1 = np.pad(microwave_omp, (0, len(dishwasher_omp) - len(microwave_omp)), mode='constant', constant_values=(0))
washerdryer_omp1 = np.pad(washerdryer_omp, (0, len(dishwasher_omp) - len(washerdryer_omp)), mode='constant', constant_values=(0))

fig, axs = plt.subplots(5, 1)
fig.set_figheight(10)
fig.set_figwidth(15)
fig.subplots_adjust(hspace=0.3)
axs[0].plot(kettle_omp1)
axs[0].set_title('Kettle')
axs[1].plot(fridgefreezer_omp1, 'tab:orange')
axs[1].set_title('Refrigerator')
axs[2].plot(microwave_omp1, 'tab:green')
axs[2].set_title('Microwave')
axs[3].plot(dishwasher_omp, 'tab:red')
axs[3].set_title('Dishwasher')
axs[4].plot(washerdryer_omp1, 'tab:purple')
axs[4].set_title('Washer Dryer')

for ax in axs.flat:
    ax.set(xlabel='Samples (1/6 Hz each index)', ylabel='Power (Watts)')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
# %%
results = pd.DataFrame(columns=['Kettle', 'Fridge', 'Microwave', 'Dishwasher', 'Washer_Dryer', 'Across_All_Appliances'], index=['MAE', 'RMSE', 'f1-Score', 'EC'])
# %%
results.Kettle = [13.98, 128.16, 0.73, 0.77]
results.Fridge = [25.11, 54.32, 0.74, 0.62]
results.Microwave = [5.66, 75.82, 0.67, 0.62]
results.Dishwasher = [19.71, 122.57, 0.67, 0.81]
results.Washer_Dryer = [23.21, 147.70, 0.67, 0.70]
results.Across_All_Appliances = [17.53, 105.71, 0.70, 0.70]
# %%
resultskettle = pd.DataFrame(columns=['MAE', 'RMSE', 'f1Score'], index=['OMP DTW', 'FHMM', 'CO'])
resultskettle.MAE = [13.98, 47.86, 71.35]
resultskettle.RMSE = [128.16, 222.83, 313.18]
resultskettle.f1Score = [0.73, 0.28, 0.16]
# %%
resultsfridge = pd.DataFrame(columns=['MAE', 'RMSE', 'f1Score'], index=['OMP DTW', 'FHMM', 'CO'])
resultsfridge.MAE = [25.11, 55.88, 76.71]
resultsfridge.RMSE = [54.32, 71.18, 114.68]
resultsfridge.f1Score = [0.74, 0.35, 0.54]
# %%
resultsmicro = pd.DataFrame(columns=['MAE', 'RMSE', 'f1Score'], index=['OMP DTW', 'FHMM', 'CO'])
resultsmicro.MAE = [5.66, 168.97, 97.66]
resultsmicro.RMSE = [75.82, 264.88, 222.46]
resultsmicro.f1Score = [0.67, 0.02, 0.05]

resultsdish = pd.DataFrame(columns=['MAE', 'RMSE', 'f1Score'], index=['OMP DTW', 'FHMM', 'CO'])
resultsdish.MAE = [19.71, 79.83, 88.42]
resultsdish.RMSE = [122.57, 258.98, 266.13]
resultsdish.f1Score = [0.67, 0.15, 0.06]

resultswasher = pd.DataFrame(columns=['MAE', 'RMSE', 'f1Score'], index=['OMP DTW', 'FHMM', 'CO'])
resultswasher.MAE = [23.21, 79.95, 116.89]
resultswasher.RMSE = [147.7, 230.24, 261.02]
resultswasher.f1Score = [0.67, 0.31, 0.10]
# %%
resultsall = pd.DataFrame(columns=['MAE', 'RMSE', 'f1Score'], index=['OMP DTW', 'FHMM', 'CO'])
resultsall.MAE = [17.53, 86.5, 90.21]
resultsall.RMSE = [105.71, 209.62, 235.49]
resultsall.f1Score = [0.70, 0.22, 0.18]
# %%
fig, axes = plt.subplots(3, 6, figsize=(20,15))
# fig.set_figheight(10)
# fig.set_figwidth(15)
axes[0, 0].set_ylabel('MAE (Watts)')
axes[1, 0].set_ylabel('RMSE (Watts)')
axes[2, 0].set_ylabel('f1-Score')
axes[0, 0].set_title('Kettle')

for i in range(6):
    axes[0, i].set_ylim([0, 175])

for i in range(6):
    axes[1, i].set_ylim([0, 350])

for i in range(6):
    axes[2, i].set_ylim([0, 1])

sns.barplot(ax=axes[0, 0], x=resultskettle.index, y=resultskettle.MAE.values, palette="Blues_d")
sns.barplot(ax=axes[1, 0], x=resultskettle.index, y=resultskettle.RMSE.values, palette="Blues_d")
sns.barplot(ax=axes[2, 0], x=resultskettle.index, y=resultskettle.f1Score.values, palette="Blues_d")

axes[0, 1].set_title('Refrigerator')
sns.barplot(ax=axes[0, 1], x=resultsfridge.index, y=resultsfridge.MAE.values, palette="Blues_d")
sns.barplot(ax=axes[1, 1], x=resultsfridge.index, y=resultsfridge.RMSE.values, palette="Blues_d")
sns.barplot(ax=axes[2, 1], x=resultsfridge.index, y=resultsfridge.f1Score.values, palette="Blues_d")

axes[0, 2].set_title('Microwave')
sns.barplot(ax=axes[0, 2], x=resultsmicro.index, y=resultsmicro.MAE.values, palette="Blues_d")
sns.barplot(ax=axes[1, 2], x=resultsmicro.index, y=resultsmicro.RMSE.values, palette="Blues_d")
sns.barplot(ax=axes[2, 2], x=resultsmicro.index, y=resultsmicro.f1Score.values, palette="Blues_d")

axes[0, 3].set_title('Dishwasher')
sns.barplot(ax=axes[0, 3], x=resultsdish.index, y=resultsdish.MAE.values, palette="Blues_d")
sns.barplot(ax=axes[1, 3], x=resultsdish.index, y=resultsdish.RMSE.values, palette="Blues_d")
sns.barplot(ax=axes[2, 3], x=resultsdish.index, y=resultsdish.f1Score.values, palette="Blues_d")

axes[0, 4].set_title('Washer Dryer')
sns.barplot(ax=axes[0, 4], x=resultswasher.index, y=resultswasher.MAE.values, palette="Blues_d")
sns.barplot(ax=axes[1, 4], x=resultswasher.index, y=resultswasher.RMSE.values, palette="Blues_d")
sns.barplot(ax=axes[2, 4], x=resultswasher.index, y=resultswasher.f1Score.values, palette="Blues_d")

axes[0, 5].set_title('Across All Appliances')
sns.barplot(ax=axes[0, 5], x=resultsall.index, y=resultsall.MAE.values, palette="Blues_d")
sns.barplot(ax=axes[1, 5], x=resultsall.index, y=resultsall.RMSE.values, palette="Blues_d")
sns.barplot(ax=axes[2, 5], x=resultsall.index, y=resultsall.f1Score.values, palette="Blues_d")



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
