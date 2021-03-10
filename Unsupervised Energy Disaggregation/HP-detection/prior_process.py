import sys
import itertools
import pickle
import time
import sys
import math
import gc
import random 

from data.data_format_eco import *

from multiprocessing import Pool
from cdtw import pydtw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import scipy.stats as scistat
import seaborn as sns
from io import StringIO
#import prettytable

from tools.rle import rlencode
from tools.handling import *
from tools.plot_tools import my_barplot

from omp.omp import *
from omp.dictionary import *
from omp.snippet4omp import *
from clust.clustering import cluster_gaussianMM

import community as co
from com.community import Community
from com.plot4community import *
from com.graph import *

from lab.labelling import *
from lab.snippet4lab import *

from perf.perf_metrics import *
from perf.formatting_perf import format_perf

from experiments.test_eco.params import *
from scipy import integrate

########5min ########
#resolution='5min'
#omp_tolerance = .06
#percentile=5
#start_date='2014-12-26'
####### 15min ######
#resolution='15min'
#omp_tolerance = .06
#percentile=10
#start_date='2014-12-24'
####### 30min ######
#resolution='30min'
#omp_tolerance = .045
#percentile=10
#start_date='2014-12-22'
####### 60min ######
resolution='60min'
omp_tolerance = .04
percentile=15
start_date='2014-12-20'
### houses from group3: IBM with HP

houseID_HP=np.array(group_participation[group_participation['group']==3]['house_number'] )

training_set=power_consumption[(power_consumption.index.get_level_values('to_ts')>start_date )&(power_consumption.index.get_level_values('to_ts')<'2015-01-01')]

training_set2=pd.DataFrame(index=training_set.index.get_level_values(1),data=training_set.values,columns=training_set.columns).resample(resolution).mean()
#training_set_15min.set_index('to_ts',append=True) 


def window(seq, n=60):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield quad(result)
    for elem in it:
        result = result[1:] + (elem,)
        yield result

global X
global TEMP
global TEMP_totDF

try:
  X = np.load( '/home/guillaume/Work/2018-07-NILM/nilm/_store/X.npy')
except FileNotFoundError:
 import omp.test_dictionary

limX = X.shape[0]

def OMP_pool(SE):
    print(SE[0])
    array_TEMP = TEMP[SE[1]:SE[2]]
    TEMP_selectDF = TEMP_totDF[SE[1]:SE[2]]

    limX = array_TEMP.shape[0]
    Xcut = X[0:limX, :]
    if any(array_TEMP!=0):
        result = OMP(x=array_TEMP, Dictionary=Xcut, dataframe=TEMP_selectDF,
                     S=200,
                     tol=omp_tolerance,
                     threshold_min_power=0)
    print('end:'+str(SE[0]))  
    return(result)

for i in range(60): # skip 51
    print(i)

    try:
        i=45
        test2=training_set[str(houseID_HP[i])].interpolate(method='linear',limit=5)
        test2=test2.fillna(0)
        test=pd.DataFrame(index=training_set.index.get_level_values(1)[1:],data=np.array([integrate.cumtrapz(test2[(i-1):(i+1)],dx=5.0) for i in range (1,3455)]).flatten()*(5/60)).resample('60min').mean()
       # test=pd.Series(index=test.index.get_level_values(1),data=test.values).resample('15min').mean()
        snippet = snippet4omp(test)
        snippet.negl_small_power(threshold=0.1)
        snippet.encoding_start_end(threshold=np.round(np.nanpercentile(test.values,10),2))
        snippet.restrict_start_end(threshold=1000)
        snippet.del_snippet_zero_power()

        start_end_omp_loop = np.transpose(np.stack((range(len(snippet.start_omp_loop)),snippet.start_omp_loop,snippet.end_omp_loop[1:])))
        print(pd.DataFrame([np.transpose(start_end_omp_loop)[0],np.transpose(start_end_omp_loop)[2]-np.transpose(start_end_omp_loop)[1]]))


        ##############################################
        ####------------ start OMP--------------- ####
        ##############################################

        #i_omp = 0  # initialise
        TEMP = np.asarray(test).flatten()
        TEMP_totDF = test


       
        pool=Pool(processes=4) 
        results=pool.map(OMP_pool,start_end_omp_loop)

        resultDic = {}
        for j in range(len(results)):
          resultDic[j]=results[j]

        path=path2store+'/'+resolution+'-kwh/houseID_{}-omp_{}-{}.pickle'.format(houseID_HP[i],omp_tolerance,resolution)
        output=open(path, 'wb')
        pickle.dump(resultDic, output)
        output.close()
        del snippet
        del test
        del TEMP_totDF
        del TEMP
        del resultDic
        del results
        gc.collect()
    except ValueError:
        print('ValueError skip: '+str(i))
    except TypeError:
        print('TypeError skip: '+str(i))
##############################################
#### ---------- GMM clustering -----------####
##############################################
#omp_tolerance = .06
gmm_output={}
#output=open(path2store+'/'+resolution+'/'+'gmm_output_omp-{}-{}.pickle'.format(omp_tolerance,resolution),'rb')
#gmm_output=pickle.load(output)
#output.close()
for i in range(60):

#for i in [45,22,18,52,56]:
    print(i)
    try:
        path=path2store+'/'+resolution+'-kwh/houseID_{}-omp_{}-{}.pickle'.format(houseID_HP[i],omp_tolerance,resolution)
        output=open(path, 'rb')
        resultDic = pickle.load(output)
        output.close()



        plt.plot(np.concatenate([resultDic[i].signal for i in resultDic.keys()])[:288]) 
        plt.plot(np.concatenate([resultDic[i].RecSignal for i in resultDic.keys()])[:288],alpha=0.75)
        plt.savefig(path2store+'/'+resolution+'-kwh/houseID_{}-raw_vs_rec-{}.pdf'.format(houseID_HP[i],resolution),bbox_inches='tight')
        plt.clf()   
        plt.close()

        number_of_components=40
        # Collect the different element from resultDic keys
        nbatoms = nbatoms_full_set(resultDic)
        Kcoef_full = Kcoef_hstack_set(Xshape=1000, resultDic=resultDic)
        ll_nbatoms = nbatoms_full_set(resultDic)

        # # --------------------- Gaussian Mixture Model --------------------- #
        gmm = cluster_gaussianMM(Kcoef_full)
        gmm.sample()
        gmm.retrieve_clus(number_component=number_of_components,random_state=100,cov='tied')
        gmm.calc_centroids()
        plot_meanCF(np.transpose(gmm.Kcoef),gmm.REFIND,ylim=[0.0,4.0],ylab='Power [kW]',xlab='time ('+resolution+')',
            title=path2store+'/'+resolution+'-kwh/houseID_{}-gmmK_{}-{}'.format(houseID_HP[i],number_of_components,resolution))

        inter_ll_atoms = np.cumsum(ll_nbatoms)
        inter_ll_atoms = np.insert(inter_ll_atoms, 0, 0)
        ll_keys = list(resultDic.keys())

        for k, l, j in zip(ll_keys, inter_ll_atoms[:-1], inter_ll_atoms[1:]):
            resultDic[k].REFIND = gmm.REFIND[l:j]

        gmm_output[houseID_HP[i]]={}
        gmm_output[houseID_HP[i]]['REFIND']=gmm.REFIND
        gmm_output[houseID_HP[i]]['centroids']=gmm.centroids

        output=open(path, 'wb')
        pickle.dump(resultDic, output)
        output.close()
    except FileNotFoundError:
        print(i)

output=open(path2store+'/'+resolution+'-kwh/gmm_output_omp-{}-{}.pickle'.format(omp_tolerance,resolution),'wb')
pickle.dump(gmm_output, output)
output.close()
