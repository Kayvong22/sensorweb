#! usr/bin/env python3
#%%
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
import math
import re
import time
import networkx as nx
import community as community_louvain
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
#%%
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


class cluster_gaussianMM(object):
    """docstring for gaussianMM."""

    def __init__(self, Kcoef):
        self.Kcoef = Kcoef
        self.X = None
        self.Z = None
        self.labels = None
        self.centroids = None
        self.lenState = None
        self.slenState = None
        self.applP = None
        self.sapplP = None
        self.gmm = None

    def sample(self):
        """Prepares the input sample from the OMP output (K strongest atoms)."""

        self.lenState = np.count_nonzero(self.Kcoef, axis=0)
        # self.lenState = 1 + 3 * np.log(self.lenState)
        self.slenState = Standardize(self.lenState)

        self.applP = np.max(self.Kcoef, axis=0)
        self.sapplP = Standardize(self.applP)

        self.X = np.vstack((self.slenState, self.sapplP*3.)).T
        # self.Z = linkage(self.X, 'ward')

    def retrieve_clus(self, number_component,random_state=None,cov='full',seed=None):
        """Retrieve the number of cluster knowing the number of clusters."""

        self.sample()

        self.gmm = GaussianMixture(n_components=number_component,covariance_type=cov,random_state=seed,warm_start=True).fit(self.X)
        self.REFIND = self.gmm.predict(self.X)

    def plotclust(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.REFIND, s=40, cmap='viridis', zorder=2)

    def calc_centroids(self):
        
        self.centroids=pd.DataFrame(np.array([np.mean(np.array([np.concatenate([[0],np.pad(np.trim_zeros(j),(0,int(999-len(np.trim_zeros(j)))),'constant')]) 
            for j in np.transpose(self.Kcoef[:,np.where(self.REFIND==i)[0]])]),axis=0) for i in np.unique(self.REFIND)]),index=np.unique(self.REFIND))

def plot_meanCF(data,clusters,ylim=[0.0,90],centroids=None,order=None,ylab='',xlab='',title=None):
    '''
    plot the mean per cluster or the centroid and the percentiles

    data: the dataset N*M as a numpy array
    centroids: centroids as a 2D numpy array N*M*K
    clusters: list of index of assignment (assigned)
    date: the date to ve written on the file
    ylim: limits of the y axis
    name: the name of the output file   
    '''                                                
    
    maxClusters=max(clusters)+1
    
    #days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    nonZeroMax=np.max(np.count_nonzero(data, axis=1))
    x=np.arange(0,nonZeroMax+2,1)
    if divmod(maxClusters,5)[1]!=0:              
        line=int(math.ceil(max(clusters)/5)+1) 
    else:
        line=int(math.ceil(int((maxClusters)/5)))
    #print('line'+str(line))
    f, axs = plt.subplots(line,5,figsize=(5*5,5*line), sharex='col', sharey='row')
    f.subplots_adjust(hspace = .2, wspace=.05)
    axs = axs.ravel()


    for i in range(maxClusters):
        #print(i)
        #axs[i].axis('off')                                    
        if order is not None:
            i2=order[i]
        else:
            i2=i

        try:
            Y = np.array([np.concatenate([[0],np.pad(np.trim_zeros(data[int(j)]),
                (0,int(nonZeroMax+1-len(np.trim_zeros(data[int(j)])))),'constant')]) for j in np.where(clusters==i2)[0]],dtype=float)

            
            if centroids is None:
                centroid=np.nanmedian(Y,axis=0)
            else:
                centroid=centroids[i2]

            if i>=(line-1)*5.:  
               # print(i)
                axs[i].set_xlabel(xlab,size=20)
                for tick in axs[i].xaxis.get_major_ticks():
                    tick.label.set_fontsize(14)   
            if i%5==0: 

                axs[i].set_ylabel(ylab,size=20)
                for tick in axs[i].yaxis.get_major_ticks():
                    tick.label.set_fontsize(14)
            axs[i].set_ylim(ylim) 

                                                             
            axs[i].set_title('Cluster: '+str(i2)+'; Size: '+str(len(Y)),size=15)


            #print(np.shape(Y))

            if len(Y)>1:
                for per in zip([2.5,5,10,25],[97.5,95,90,75]):
                    axs[i].fill_between(x,np.nanpercentile(Y,per[0],axis=0),np.nanpercentile(Y,per[1],axis=0), color="#3F5D7D",alpha=0.3)                                                            
                axs[i].plot(x,centroid,lw=1, color="white")
            else:
                axs[i].plot(x,centroid,lw=1, color="#3F5D7D")
        except ValueError:
            print('skip: '+str(i))
    #day= datetime.strptime(date,'%Y-%m-%d').weekday() 
    #f.suptitle('date = '+date+' ('+days[day]+')',size=15)          
    if title==None: 
        f.savefig('./gmm-K'+ str(maxClusters) +'.png',bbox_inches='tight') 
    else: 
        f.savefig(title +'.png',bbox_inches='tight')
    plt.clf()   
    plt.close()

def Kcoef_hstack_set(Xshape, resultDic):
    """Returns a stack set of the strong atoms.

    Args:
        -- Xshape: longest atom in the OMP dictionary
        -- resultDic: collection of all result over the steps
    """
    newKcoef_dict = {}
    for i in list(resultDic.keys()):
        #print(i)
        #print(resultDic[i]==None)
        addzeros = Xshape - resultDic[i].Kcoef.shape[0]
        newKcoef_dict[i] = np.pad(
            resultDic[i].Kcoef, ((0, addzeros), (0, 0)), mode='constant')

    Kcoef_hstack = np.hstack([newKcoef_dict[i]
                              for i in resultDic.keys()])

    return Kcoef_hstack


class Community():
    """Compute the community detection through Louvain method."""

    def __init__(self, Kcoef, REFIND, nbatoms):
        # super(Community, self).__init__()

        self.Kcoef = Kcoef
        self.REFIND = REFIND
        self.nbatoms = nbatoms

        self.sample_graph = None
        self.partition = None
        self.p = None
        self.ComDict = None

    def do_Com2Dict(self, resultDic):
        """Insert the community detection output in the 'resultDic' function"""

        self.partition = self.p
        self.p = self.partition.item()
        ll_communities = list(self.p.keys())

        for k in resultDic.keys():
            resultDic[k].ComDict = {}
            for com in ll_communities:
                ll_p = [int(ppp) for ppp in self.p[com]]  # from str to int
                ind4Kcoef = [i for i, ll in enumerate(
                    resultDic[k].REFIND.tolist()) if int(ll) in ll_p]
                resultDic[k].ComDict[com] = resultDic[k].Kcoef[:, ind4Kcoef]

    def _build_SampleGraph(self):
        self.sample_graph = nx.Graph()

        df_tuples = [tuple(x) for x in self.df.to_records(index=False)]
        self.sample_graph.add_weighted_edges_from(df_tuples)
#%%
x = np.loadtxt('/Users/kayvon/Documents/Engineering/Disaggregation/New/house_3/channel_1.dat')
# try 450000:465000
#%%
# x1 = np.loadtxt('/Users/kayvon/Documents/Engineering/Disaggregation/ukdale/house_3/channel_1.dat',
#     usecols=(0,1), skiprows=6675, max_rows=1000)

#%%
# x = open_pickle('/Users/kayvon/Desktop/synthetic_data.pkl')
# x1 = x['kettle'] + x['fridge'] + x['microwave'] + x['washingmachine'] + x['dishwasher']
dictionary2 = open_pickle('/Users/kayvon/Desktop/dictionary.pkl')
# boxcarinfos = open_pickle('/Users/kayvon/Documents/Engineering/Disaggregation/boxcarinfos.pkl')
######################
# try less than 1000 -- Doesn't work
#####################
#%%
def run_omp(data,dictionary):
    '''
    Function to run OMP on a portion of data

    Arguments:
        data: input signal you want to be approximated
        dictionary: overcomplete dictionary of boxcars
    '''
    start_time = time.time()
    df = pd.DataFrame()
    result = OMP(x = data[:,1], Dictionary=dictionary,dataframe=df,
    maxiter=5000,tol=0.0225,S=200,threshold_min_power=0)
    print('Result finished.')
    print('--- %s seconds ---' % (time.time() - start_time))
    print('--- %s minutes ---' % ((time.time() - start_time)/60))

    return result

def time_period(data,start,stop):
    ts1 = int(data[start,0])
    ts2 = int(data[stop,0])
    # x = datetime.utcfromtimestamp(ts) ### THIS GIVES *datetime.datetime(2013, 2, 27, 20, 35, 20)*
    print('Data from time:')
    print(datetime.utcfromtimestamp(ts1).strftime('%Y-%m-%d %H:%M:%S'))
    print('to time:')
    print(datetime.utcfromtimestamp(ts2).strftime('%Y-%m-%d %H:%M:%S'))
    print(' ')

def make_resultDic(data, dictionary,start,stop,time_info):
    '''
    Function to run OMP on multiple batches of data

    Arguments:
        data: timestamps in column 0 and input signal you want to be approximated in column 1
        dictionary: overcomplete dictionary of boxcars
        start: index of where you want to start in 'data'
        stop: index of where you want to stop in 'data'
            stop should be in an increment of (start+x(1000))+1 
            where x is how many batches you want
            Ex: start = 1000
                stop = 4001
                will give OMP of data[1000:2000] then data[2000:3000] and so on
                the last OMP will be OMP of data[4000:5000]
        time_info: True or False -- if the timestamp is included in the 'data' variable column 0,
            this will give you the timestamp for the entire result start to stop
    '''
    start_time_total = time.time()
    resultDic = {}
    cnt = 0
    
    if time_info==True: 
        time_period(data,start,stop)
        
    # time_period(data,start,stop)
    # for z in range(int(((stop+999) - start)/1000)):
    #         resultDic[z] = []
    for i in np.arange(start,stop,1000):
        j = i + 1000
        data_partial = data[i:j]
        # print(i)
        result = run_omp(data_partial, dictionary)
        resultDic[cnt] = result
        cnt += 1

    print(' ')
    print('--- %s minutes for total ---' % ((time.time() - start_time_total)/60))
    return resultDic

#%%
resultDic = make_resultDic(x,dictionary2,450000,480001,time_info=True) # 450000,499001
#%%
# start_time1 = time.time()
# df1 = pd.DataFrame()
# result1 = OMP(x = x1[:,1], Dictionary=dictionary2,dataframe=df1,
#     maxiter=5000,tol=0.055,S=200,threshold_min_power=0)
# print('result1 finished')
# print('--- %s seconds ---' % (time.time() - start_time1))
# print('--- %s minutes ---' % ((time.time() - start_time1)/60))

#%%
save_pickle(resultDic,'/Users/kayvon/Documents/Engineering/Disaggregation/resultDic50.pkl')

#%%
Kcoef_full = Kcoef_hstack_set(Xshape=1000, resultDic=resultDic)

#%%
number_of_clusters = 5

gmm = cluster_gaussianMM(Kcoef_full)
gmm.sample()
gmm.retrieve_clus(number_component=number_of_clusters,random_state=100,cov='tied')
gmm.plotclust()
gmm.calc_centroids()

plot_meanCF(np.transpose(gmm.Kcoef),gmm.REFIND,ylim=[0.0,4000.0],
    ylab='Power [kW]',xlab='time',title=None)
#%%
def nbatoms_full_set(resultDic):
    """Returns the entire set of number of atom on each sequence k (resultDic[k])."""
    ll_nbatoms = []
    for i in list(resultDic.keys()):
        ll_nbatoms.append(resultDic[i].nbatoms)

    return ll_nbatoms

ll_nbatoms = nbatoms_full_set(resultDic)

inter_ll_atoms = np.cumsum(ll_nbatoms)
inter_ll_atoms = np.insert(inter_ll_atoms, 0, 0)
ll_keys = list(resultDic.keys())
for k, l, j in zip(ll_keys, inter_ll_atoms[:-1], inter_ll_atoms[1:]):
            resultDic[k].REFIND = gmm.REFIND[l:j]

gmm_output={}
gmm_output['REFIND']=gmm.REFIND
gmm_output['centroids']=gmm.centroids

# %%
def Kcoef_ts_set(resultDic, Xshape):
    """Returns a stack set of the strong atoms projected on the full time
    series length.

    Args:
        -- resultDic: collection of all result over the steps
    """
    ll = []

    for k in resultDic.keys():

        #print(k)
        ll.append(len(resultDic[k].signal))

    tot_length = np.cumsum(np.asarray(ll))[-1]
    ll_cumsum = np.cumsum(ll)

    start_seq_k = np.insert(ll_cumsum, 0, 0)[:-1]
    end_seq_k = np.asarray(ll_cumsum)

    Kcoef_hstack = Kcoef_hstack_set(Xshape=Xshape, resultDic=resultDic)

    shape = (tot_length, Kcoef_hstack.shape[1])
    print(shape)
    mat_Kcoef = np.empty(shape)

    i_mat = 0
    for k, i, j in zip(resultDic.keys(), start_seq_k, end_seq_k):
        for i_Kcoef in range(resultDic[k].Kcoef.shape[1]):
            mat_Kcoef[i:j,
                      i_mat] = resultDic[k].Kcoef[:, i_Kcoef]
            i_mat += 1

    return mat_Kcoef


def REFIND_full_set(resultDic):
    """Returns the entire set of labels for all the strong atoms."""

    REFIND_full = []
    for i in list(resultDic.keys()):
        REFIND_full.append(resultDic[i].REFIND.tolist())

    flat_list = [item for sublist in REFIND_full for item in sublist]
    REFIND_full = np.asarray(flat_list)

    return REFIND_full
#%%
class Graph(object):
    """Compute the edges, nodes and weights of each and every atoms.."""

    def __init__(self, resultDic, l1lim, sample_per_min, ind_stay, Xshape):
        self.resultDic = resultDic
        self.threslhold_l1_distance = (sample_per_min * 1440 * (1 - l1lim))
        self.ind_stay = ind_stay
        self.Xshape = Xshape

    def atom_position(self):
        """ ."""

        self.mat_Kcoef = Kcoef_ts_set(self.resultDic, Xshape=self.Xshape)

        self.absolute_atoms_middle = np.empty((self.mat_Kcoef.shape[1],))
        for k in range(self.mat_Kcoef.shape[1]):
            self.absolute_atoms_middle[k] = (np.nonzero(self.mat_Kcoef[:, k])[0][0] +
                                             np.nonzero(self.mat_Kcoef[:, k])[0][-1]) / 2
        print(len(self.absolute_atoms_middle))

    def alterego(self):
        """ ."""

        self.absolute_atoms_middle = self.absolute_atoms_middle[self.ind_stay]

        self.DF_graph = pd.DataFrame(columns=['alter', 'ego', 'link'])

        REFIND_full = REFIND_full_set(self.resultDic)
        REFIND_stay = REFIND_full[self.ind_stay]

        for i in range(len(self.absolute_atoms_middle)):
            absolute_distance_atom_k = abs(self.absolute_atoms_middle[i] - self.absolute_atoms_middle)
            REFIND_close_k = REFIND_stay[np.where((absolute_distance_atom_k < self.threslhold_l1_distance) &
                                        (absolute_distance_atom_k != float(0)))[0]]

            alter = REFIND_stay[i]
            ego = REFIND_close_k

            alter = np.tile(alter, len(ego))

            TEMPdf = pd.DataFrame({
                'alter': alter.astype(int),
                'ego': ego.astype(int),
                'link': np.tile(1, len(ego))
            })

            self.DF_graph = pd.concat([self.DF_graph, TEMPdf])

        self.DF_graph = self.DF_graph.reset_index(drop=True)
        return self.DF_graph

#%%
def atom_position(resultDic,Xshape):
    """ ."""

    mat_Kcoef = Kcoef_ts_set(resultDic, Xshape=Xshape)

    absolute_atoms_middle = np.empty((mat_Kcoef.shape[1],))
    for k in range(mat_Kcoef.shape[1]):
        absolute_atoms_middle[k] = (np.nonzero(mat_Kcoef[:, k])[0][0] +
                                            np.nonzero(mat_Kcoef[:, k])[0][-1]) / 2
    return absolute_atoms_middle
#%%
absatoms = atom_position(resultDic,Xshape=1000)
# %%
G = nx.Graph()
for i in absatoms:
    # print(i)
    #all_graphs[i] = nx.Graph()
    G.add_node(i)
#%%
nodes_df = pd.DataFrame({'nodes':absatoms})

cluster = {} 
for i in absatoms:
    # print(i)
    inner_arr = []
    for j in absatoms:
        inner_arr.append(w_uv(i,j))

    cluster[i] = inner_arr
out = pd.concat([nodes_df, pd.DataFrame(cluster)], axis=1)

#%%

#%%
#%%
# temp = resultDic[1].Kcoef[:,1]

# tempX = np.ma.masked_equal(temp,0)

# where = np.where(tempX == np.max(tempX))

# myx = statistics.median(where[0])

# plt.plot(resultDic[1].Kcoef[:,1])
# plt.axvline(x=myx)

# %%
# for i in range(np.shape(resultDic[j].Kcoef)[1]):
# #for i in range(len(clusters)):
#     #print(i)
#     temp = 0
#     tempX = 0
#     where = 0
#     myx = 0
#     temp = resultDic[j].Kcoef[:,i]
#     tempX = np.ma.masked_equal(temp,0)
#     where = np.where(tempX == np.max(tempX))
#     myx = statistics.median(where[0])
#     print(myx)
#     # plt.plot(resultDic[j].Kcoef[:,i])
#     # plt.axvline(x=myx,color='red')

# %%
clusters = pd.DataFrame(Kcoef_full.transpose())
clusters1 = pd.DataFrame(gmm.centroids)
clusters['label'] = gmm.REFIND
#%%
cluster = {}
finaldict = {}

for i in range(number_of_clusters):
    lab = 'grp' + str(i)
    tempdf = clusters[clusters['label']==i]
    
    cluster[lab] = tempdf
    cluster[lab] = cluster[lab].reset_index()

for i in cluster.keys():
    cluster[i] = cluster[i].transpose().drop(['index','label'], axis=0)
#%%
for i in cluster.keys():  
    #print(i)
    finaldict[i] = []
    for j in range(np.shape(cluster[i])[1]):
        print(j)
        temp = 0
        tempX = 0
        where = 0
        myx = np.array([])
        temp = cluster[i][j]
        tempX = np.ma.masked_equal(temp,0)
        where = np.where(tempX == np.max(tempX))
        myx1 = median(where[0])
        myx = np.append(myx,myx1)
        print(myx)
        finaldict[i] = np.append(finaldict[i],myx)
#%%


for i in range(np.shape(clusters1)[0]):
    finaldict[i] = []
    for j in range(np.shape(clusters1)[1]):
        # print(j)
        temp = 0
        tempX = 0
        where = 0
        myx = np.array([])
        temp = clusters1[i][j]
        tempX = np.ma.masked_equal(temp,0)
        where = np.where(tempX == np.max(tempX))
        myx1 = median(where[0])
        myx = np.append(myx,myx1)
        print(myx)
        finaldict[i] = np.append(finaldict[i],myx)

#%%
### NEW ###
finaldict = {}
for i in range(len(clusters1)):
    temp = 0
    tempX = 0
    where = 0
    myx = np.array([])
    temp = clusters1.iloc[i,:]
    tempX = np.ma.masked_equal(temp,0)
    where = np.where(tempX != 0)
    myx1 = median(where[0])
    myx = np.append(myx,myx1)
    print(i)
    print(myx)
    finaldict[i] = myx

#%%
all_graphs = {}
for i in finaldict.keys():
    all_graphs[i] = nx.Graph()
    for j in range(len(finaldict[i])):
        all_graphs[i].add_node(finaldict[i][j])
#%%
### NEW ###
#all_graphs = {}
G = nx.Graph()
for i in finaldict.keys():
    #all_graphs[i] = nx.Graph()
    G.add_node(finaldict[i][0])
# %%
def w_uv(tu,tv):
    """ Calculate weights for edges between nodes

    Arguments:
        tu and tv are respectively the position of the 
        center of boxcar functions u and v.
    """
    w = math.exp((-(tu-tv)**2)/(2*(0.95**2)))
    return w

def create_df_weight_cols(grp):
    """ Creates pandas DataFrame with nodes as indices 
        and the weights between nodes filling the matrix
    
    Arguments:
        grp: all_grps['label'] -> 'label' is written as grp0 to grp43
    
    Returns:
        pd.DataFrame
    """
    nodes_df = pd.DataFrame({'nodes':grp})

    cluster = {} 
    for i in grp:
        inner_arr = []
        for j in grp:
            inner_arr.append(w_uv(i,j))

        cluster[i] = inner_arr
    return pd.concat([nodes_df, pd.DataFrame(cluster)], axis=1)
#%%
# Creates DataFrames of weights between nodes for each cluster group from GMM
all_dfs = {}
for i in finaldict.keys():
    all_dfs[i] = create_df_weight_cols(finaldict[i])
#%%
### NEW ###
nodes_df = pd.DataFrame({'nodes':finaldict})
#%%
cluster = {} 
for i in finaldict:
    print(i)
    inner_arr = []
    for j in finaldict:
        inner_arr.append(w_uv(i,j))

    cluster[i] = inner_arr
out = pd.concat([nodes_df, pd.DataFrame(cluster)], axis=1)
# %%
def add_edges_with_weights(grp,df,all_graphs):
    """ Adds edges between nodes in NetworkX graph 
        and assigns the weight calculated to that edge

        Arguments:
            grp: all_grps['label'] dict - Contains indices of cluster groups
            df: all_dfs['label'] dict - Contains all weights between nodes/indices
            lab: str 'label' -> 'label' is written as grp0 to grp43
            all_graphs: all_graphs['label'] dict - Contains NetworkX graphs
    """
    c = 0
    for i in grp:
        for j in grp[c+1:]:
            print(i)
            print(j)
        #     weights = df[df['nodes']==i][j].values[0]
        #     all_graphs.add_edge(i,j,weight=weights)
        # c += 1
    return

#%%

# THIS ONE
c = 0
for i in absatoms:
    for j in absatoms[c+1:]:
        # print(i)
        # print(j)
        weights = out[out['nodes']==i][j].values[0]
        if weights != 0 and weights != 1:
            G.add_edge(i,j,weight=weights)
        else:
            continue
# %%
for i in all_graphs.keys():
    add_edges_with_weights(finaldict[i],all_dfs[i],all_graphs[i])
# %%
# Using python-louvain package allows to find best communities(partition) for nodes
for i in all_graphs.keys():
    partition = community_louvain.best_partition(all_graphs[i])
    all_dfs[i]['partition'] = all_dfs[i].nodes.apply(lambda node: partition[node])
# %%
partition = {}
for i in all_graphs.keys():
    partition[i] = community_louvain.best_partition(all_graphs[i])
#%%
lastdict = {} 
for i in range(number_of_clusters):
    lastdict[i] = clusters[clusters['label']==i]
for j in all_graphs.keys():
    for i in range(number_of_clusters):
        lastdict[i].append(pd.Series(partition[i].values()),ignore_index=True)


#%%
partition = community_louvain.best_partition(G)
partition_df = pd.DataFrame(partition,index=partition)
partition_df = pd.DataFrame(partition_df.iloc[0,:].transpose())
partition_df = partition_df.reset_index()
partition_df = partition_df.rename(columns={617.0: 'partition'})
#%%
partition_df = partition_df.sort_values(by=['partition'])
# %%
groupzero = partition_df[partition_df['partition']==0]
# %%
groupsix = partition_df[partition_df['partition']==6]
# %%
groupseven = partition_df[partition_df['partition']==7]
#%%
groupzero.append(pd.Series(partition.values()),ignore_index=True)
# %%
final = groupzero[groupzero['partition']==9]
# %%
