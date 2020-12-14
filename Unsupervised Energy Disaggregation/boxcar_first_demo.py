#!/usr/bin/env python3

## Unsupervised Energy Disaggregation ##
## Author: Kayvon Ghahremani

#%%
from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import re
import time
import networkx as nx
import community as community_louvain
import matplotlib.cm as cm

#%%
# Functions defined to create boxcar dictionary
def Heaviside(x, a):
    """Compute a Heaviside function."""
    y = (np.sign(x - a) + 1) / 2
    y[y == 0.5] = 0
    return y


def Boxcar(l, w, t):
    """Compute a boxcar function.

    Arguments:
        l -- constant in the interval (non-zero)
        w -- width of the boxcar, i.e. the interval equal to constant 'l'
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
        infos: if 'True' return the information about the boxcar
        boxwidth

    For now, synthetic fridge and heat pump over 140 minutes.
    """

    # Time vector
    x = np.linspace(1, tslength, tslength)

    # Dictionary matrix
    X1 = np.eye(tslength)
    X2 = np.zeros((tslength, tslength), dtype=float)
    X = np.concatenate((X1, X2), axis=1)

    if tslength < boxwidth:
        boxwidth = tslength

    boxcarinfos = np.zeros((tslength * boxwidth, 2), dtype=float)

    for j in range(1, boxwidth):
        for i in range(1, tslength):
            X2[:, i] = Boxcar(i, j, x) #(l,w,t)
            boxcarinfos[j,0] = j
            boxcarinfos[i,1] = i
        X = np.concatenate((X, X2), axis=1)

    if not infos:
        return X

    else:
        return X, boxcarinfos

#%%
# To save and load pkl files
import dill

def save_pickle(obj,file_name):
    with open(file_name, 'wb') as handle:
        dill.dump(obj, handle, protocol=dill.HIGHEST_PROTOCOL)
    return file_name + ' is saved'

def open_pickle(file_name):
    with open(file_name, 'rb') as handle:
        obj = dill.load(handle)
    return obj

#%%
#%% 
# Load data of house 3 - UK_DALE Dataset
# 'channel_1.dat' is the aggregate signal
x = np.loadtxt('/Users/Kayvon/Desktop/channel_1.dat',usecols=(0,1), max_rows=1000)

#%%
##### DO NOT RUN THIS - It will take hours to finish #########
# I have the 'dictionary2' variable pickled if you want to run it
dictionary2, boxcarinfos = gen_dict2(tslength=1000,boxwidth=1000,infos=True)

# save_pickle(dictionary2, '/Users/Kayvon/Desktop/dictionary2.pkl')
# save_pickle(boxcarinfos, '/Users/Kayvon/Desktop/boxcarinfos.pkl')

#%%
# Compute OMP - Power Signal Sparse Approximation
omp = OrthogonalMatchingPursuit()
omp.fit(X=dictionary2,y=x[:,1])
coef = omp.coef_
idx_r, = coef.nonzero()

#%%
from sklearn.mixture import GaussianMixture as GMM

gmm = GMM(n_components=43,random_state=100,covariance_type='tied').fit(idx_r.reshape(-1,1))
labels = gmm.predict(idx_r.reshape(-1,1))

#%%
# This organizes each cluster from GMM into a dict for easier access
all_grps = {}
for i in range(0,43):
    lab = "grp" + str(i)
    all_grps[lab] = idx_r[labels==i]

# This creates a graph using NetworkX for each cluster
all_graphs = {}
for i in range(0,43):
    lab = 'grp' + str(i)
    all_graphs[lab] = nx.Graph()
    for j in range(len(all_grps[lab])):
        all_graphs[lab].add_node(all_grps[lab][j])

#%%
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
for i in range(0,43):
    lab = "grp" + str(i)
    all_dfs[lab] = create_df_weight_cols(all_grps[lab])

#%%
def add_edges_with_weights(grp,df,lab,all_graphs):
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
            weights = df[df['nodes']==i][j].values[0]
            all_graphs[lab].add_edge(i,j,weight=weights)
        c += 1
    return

# Creates edges between nodes and assignes weights to edges
for i in all_graphs.keys():
        add_edges_with_weights(all_grps[i],all_dfs[i],i,all_graphs)


#%%
# Using python-louvain package allows to find best communities(partition) for nodes
for i in all_graphs.keys():
    partition = community_louvain.best_partition(all_graphs[i])
    all_dfs[i]['partition'] = all_dfs[i].nodes.apply(lambda node: partition[node])

#%%
def get_nodes_by_partition(df):
    """ Pulls indices of nodes within the same community(partition)

        Arguments:
            df: all_dfs['label'] -> 'label' is written as grp0 to grp43

        Returns: 
            Dict of nodes organized by community(partition)
    """
    temp_dict = {}
    for p in df.partition.value_counts().keys():
        temp_dict[p] = df[df['partition']==p].nodes.to_list()

    return temp_dict

# Creates dict of nodes by community
partitions_communities_dict = {}
for i in all_dfs.keys():
    partitions_communities_dict[i] = get_nodes_by_partition(all_dfs[i])

#%%
# Load real data for the appliances in house 3
Kettle = np.loadtxt('/Users/Kayvon/Desktop/channel_2.dat', usecols=(0,1),max_rows=1000)
ElectricHeater = np.loadtxt('/Users/Kayvon/Desktop/channel_3.dat', usecols=(0,1),max_rows=1000)
Laptop = np.loadtxt('/Users/Kayvon/Desktop/channel_4.dat', usecols=(0,1),max_rows=1000)
Projector = np.loadtxt('/Users/Kayvon/Desktop/channel_5.dat', usecols=(0,1),max_rows=1000)

###############################################################################################
# After all of this, each index of each community(partition) is found in the 'dictionary2' 
# from the beginning. Then they are plotted as communities and visually compared to the 
# real signals found just above. I would like to implement a trained-with-labels model that 
# contains training from the actual appliance data, and with this the model will match the 
# boxcar functions found in each community for me.
