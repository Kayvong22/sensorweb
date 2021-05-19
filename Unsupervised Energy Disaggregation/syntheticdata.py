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
# %%
import dill
def save_pickle(obj,file_name):
    with open(file_name, 'wb') as handle:
        dill.dump(obj, handle, protocol=dill.HIGHEST_PROTOCOL)
    return file_name + ' is saved'

def open_pickle(file_name):
    with open(file_name, 'rb') as handle:
        obj = dill.load(handle)
    return obj

# Get activation examples using nilmtk
import nilmtk
from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.dataset_converters import convert_ukdale
from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.legacy.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

from matplotlib import rcParams
import matplotlib.pyplot as plt
import math
import h5py
import numpy as np
import pandas as pd
import time
from datetime import datetime

dataset = DataSet('/Users/kayvon/Downloads/ukdale.h5')
dataset.set_window(start="1-1-2013",end="31-12-2013")
BUILDING = 1
elec = dataset.buildings[BUILDING].elec
activations = {}
microwave = elec['microwave']
activations['microwave'] = microwave.get_activations(on_power_threshold=200,min_on_duration=12,min_off_duration=30)
microwave = elec['kettle']
activations['kettle'] = microwave.get_activations(on_power_threshold=2000,min_on_duration=12,min_off_duration=0)
microwave = elec['fridge freezer']
activations['fridge freezer'] = microwave.get_activations(on_power_threshold=50,min_on_duration=60,min_off_duration=12)
microwave = elec['dish washer']
activations['dish washer'] = microwave.get_activations(on_power_threshold=10,min_on_duration=1800,min_off_duration=1800)
microwave = elec['washer dryer']
activations['washer dryer'] = microwave.get_activations(on_power_threshold=20,min_on_duration=1800,min_off_duration=160)

# After plotting through some of the activations these are the ones I chose for synthetic data
washerdryer = activations['washer dryer'][3].to_list()
dishwasher = activations['dish washer'][18].to_list()
fridgefreezer = activations['fridge freezer'][6].to_list()
kettle = activations['kettle'][4].to_list()
microwave = activations['microwave'][12].to_list()

# Dictionary built using gen_dict w/ some modifications
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

# %%
result = {}
for i in omp_dict.keys():
    df = pd.DataFrame()
    result[i] = OMP(x=omp_dict[i], Dictionary=dictionary, dataframe=df, maxiter=5000, tol=0.055, S=200, threshold_min_power=0)
# %%
washerdryer_omp = result['washerdryer'].RecSignal[0:len(appliance_dict['washerdryer'])]
dishwasher_omp = result['dishwasher'].RecSignal[0:len(appliance_dict['dishwasher'])]
fridgefreezer_omp = result['fridgefreezer'].RecSignal[0:len(appliance_dict['fridgefreezer'])]
kettle_omp = result['kettle'].RecSignal[0:len(appliance_dict['kettle'])]
microwave_omp = result['microwave'].RecSignal[0:len(appliance_dict['microwave'])]
# %%
omp_smooth = {}
omp_smooth['washerdryer'] = washerdryer_omp
omp_smooth['dishwasher'] =  dishwasher_omp
omp_smooth['fridgefreezer'] =  fridgefreezer_omp
omp_smooth['kettle'] = kettle_omp
omp_smooth['microwave'] = microwave_omp
# %%
agg =  np.zeros(dictionary.shape[0])

for i in omp_smooth.keys():
    num = np.random.randint(2000-len(omp_smooth[i]))
    if i == 'washerdryer':
        num = np.random.randint(2000-len(omp_smooth[i]))
        agg[num:num+omp_smooth[i].shape[0]] += omp_smooth[i]
        # print(False)
    elif i == 'dishwasher':
        num = np.random.randint(2000-len(omp_smooth[i]))
        agg[num:num+omp_smooth[i].shape[0]] += omp_smooth[i]
        # print(False)
    elif i == 'fridgefreezer':
        for j in range(3):
            num = np.random.randint(2000-len(omp_smooth[i]))
            agg[num:num+omp_smooth[i].shape[0]] += omp_smooth[i]
    elif i == 'kettle':
        for k in range(2):
            num = np.random.randint(2000-len(omp_smooth[i]))
            agg[num:num+omp_smooth[i].shape[0]] += omp_smooth[i]
    elif i == 'microwave':
        for z in range(2):
            num = np.random.randint(2000-len(omp_smooth[i]))
            agg[num:num+omp_smooth[i].shape[0]] += omp_smooth[i]
            
df1 = pd.DataFrame()
result_final = OMP(x=agg, Dictionary=dictionary, dataframe=df1, maxiter=10000, tol=0.002, S=1000, threshold_min_power=0)


# Here filter out type 1 (just on/off; fridge, microwave, kettle) appliances by dynamic time warping
from tslearn.metrics import dtw, soft_dtw

kettle_index_list = []
for i in range(result_final.Kcoef.shape[1]):
    dtw_score_kettle = dtw(result_final.Kcoef[:,i], kettle_omp)
    print('DTW Kettle Score ' + str(i) + '= ' +str(dtw_score_kettle))
    if dtw_score_kettle < 5:
        kettle_index_list.append(i)
kettle_index_list
# %%
microwave_index_list = []
for i in range(result_final.Kcoef.shape[1]):
    dtw_score_microwave = dtw(result_final.Kcoef[:,i], microwave_omp)
    print('DTW Microwave Score ' + str(i) + '= ' +str(dtw_score_microwave))
    if dtw_score_microwave < 5:
        microwave_index_list.append(i)
microwave_index_list
# %%
fridgefreezer_index_list = []
for i in range(result_final.Kcoef.shape[1]):
    dtw_score_fridgefreezer = dtw(result_final.Kcoef[:,i], fridgefreezer_omp)
    print('DTW Fridge Freezer Score ' + str(i) + '= ' +str(dtw_score_fridgefreezer))
    if dtw_score_fridgefreezer < 5:
        fridgefreezer_index_list.append(i)
fridgefreezer_index_list
# %%
fig, ax = plt.subplots()

for j in kettle_index_list:
    ax.plot(result_final.Kcoef[:,j], color='blue', 
    label='Kettle' if j == kettle_index_list[0] else "")
for i in microwave_index_list:
    ax.plot(result_final.Kcoef[:,i], color='red', 
    label='Microwave' if i == microwave_index_list[0] else "")
for k in fridgefreezer_index_list:
    ax.plot(result_final.Kcoef[:,k], color='green', 
    label='Fridgefreezer' if k == fridgefreezer_index_list[0] else "")

plt.ylim([0, 4000])
plt.legend()
plt.show()

# Separate the classified type 1 aplliances from the rest of the OMP chosen boxcars
# See whats left over (washer dryer and dishwasher)
# %%
indexes = np.arange(0,16).tolist()
type1_indexes = []
for i in range(len(kettle_index_list)):
    type1_indexes.append(kettle_index_list[i])
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
