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
# %%
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


def norm2(x):
    """Compute a l2-norm."""
    return np.linalg.norm(x) / np.sqrt(len(x))


def gen_dict(tslength, boxwidth=120):
    """Generate a dictionary.

    Arguments:
        tslength: time series length
        infos: if 'True' return the information about the boxcar
        boxwidth

    For now, synthetic fridge and heat pump over 140 minutes.
    """

    x = np.linspace(1, tslength, tslength)

    if tslength < boxwidth:
        boxwidth = tslength

    ll = []
    for j in range(1, boxwidth):
        # print(j)
        for i in range(1, tslength):
            ll.append(Boxcar(i, j, x))

    return np.array([mm for mm in ll]).T

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
#%% house 2
x = np.loadtxt('/Users/komranghahremani/Desktop/channel_1_2.dat',usecols=(0,1), max_rows=1000)
#%% house 3
x = np.loadtxt('/Users/komranghahremani/Desktop/channel_1.dat',usecols=(0,1), max_rows=1000)
# %%
dictionary2, boxcarinfos = gen_dict2(tslength=1000,boxwidth=1000,infos=True)
# %%
save_pickle(dictionary2, '/Users/komranghahremani/Desktop/dictionary2.pkl')
save_pickle(boxcarinfos, '/Users/komranghahremani/Desktop/boxcarinfos.pkl')
# %%
# Compute OMP
omp = OrthogonalMatchingPursuit()
omp.fit(X=dictionary2,y=x[:,1])
coef = omp.coef_
idx_r, = coef.nonzero() #idx_r is the nonzero parts of coef
# %%
from sklearn.mixture import GaussianMixture as GMM

gmm = GMM(n_components=43,random_state=100,covariance_type='tied').fit(idx_r.reshape(-1,1))

# gmm = GMM(n_components=43).fit(idx_r.reshape(-1,1))
labels = gmm.predict(idx_r.reshape(-1,1))
#%%
all_grps = {}

for i in range(0,43):
    lab = "grp" + str(i)
    all_grps[lab] = idx_r[labels==i]


all_graphs = {}
for i in range(0,43):
    lab = 'grp' + str(i)
    all_graphs[lab] = nx.Graph()
    for j in range(len(all_grps[lab])):
        all_graphs[lab].add_node(all_grps[lab][j])

#%%
def w_uv(tu,tv):
    w = math.exp((-(tu-tv)**2)/(2*(0.95**2)))
    return w

def create_df_weight_cols(grp):

    nodes_df = pd.DataFrame({'nodes':grp})

    cluster = {} 
    for i in grp:
        inner_arr = []
        for j in grp:
            inner_arr.append(w_uv(i,j))

        cluster[i] = inner_arr
    return pd.concat([nodes_df, pd.DataFrame(cluster)], axis=1)

all_dfs = {}
for i in range(0,43):
    lab = "grp" + str(i)
    all_dfs[lab] = create_df_weight_cols(all_grps[lab])

#%%
def add_edges_with_weights(grp,df,lab,all_graphs):
    c = 0
    for i in grp:
        for j in grp[c+1:]:
            weights = df[df['nodes']==i][j].values[0]
            all_graphs[lab].add_edge(i,j,weight=weights)
        c += 1
    return

for i in all_graphs.keys():
        add_edges_with_weights(all_grps[i],all_dfs[i],i,all_graphs)

#%%
for i in all_graphs.keys():
    partition = community_louvain.best_partition(all_graphs[i])
    all_dfs[i]['partition'] = all_dfs[i].nodes.apply(lambda node: partition[node])

#%%
def get_nodes_by_partition(df):
    temp_dict = {}
    for p in df.partition.value_counts().keys():
        temp_dict[p] = df[df['partition']==p].nodes.to_list()

    return temp_dict

partitions_communities_dict = {}
for i in all_dfs.keys():
    partitions_communities_dict[i] = get_nodes_by_partition(all_dfs[i])

#%%
# for i in partitions_communities_dict.keys():
#     partitions_communities_dict[i][]
# %% House 3
# grp0 -> W
plt.plot(dictionary2[:,4501])
plt.plot(dictionary2[:,4517])
plt.plot(dictionary2[:,4553])

plt.plot(dictionary2[:,4440])
plt.plot(dictionary2[:,4451])
# %% grp1 -> L
plt.plot(dictionary2[:,966499])
# %% grp2 -> possible 
plt.plot(dictionary2[:,146640])
#%% grp3 -> possible
plt.plot(dictionary2[:,16133])
#%% grp4 -> short lines (fridge?)
plt.plot(dictionary2[:,720],c='r')
plt.plot(dictionary2[:,732],c='r')
plt.plot(dictionary2[:,739],c='r')

plt.plot(dictionary2[:,680],c='b')
plt.plot(dictionary2[:,684],c='b')
plt.plot(dictionary2[:,687],c='b')

plt.plot(dictionary2[:,652],c='g')
plt.plot(dictionary2[:,655],c='g')
plt.plot(dictionary2[:,662],c='g')

plt.plot(dictionary2[:,693],c='c')
plt.plot(dictionary2[:,659],c='c')

plt.plot(dictionary2[:,646],c='y')
plt.plot(dictionary2[:,648],c='y')

plt.plot(dictionary2[:,625],c='k')
plt.plot(dictionary2[:,633],c='k'
# %% grp5 -> possible (short line)
plt.plot(dictionary2[:,30628],c='b')

# %% grp6 -> L
plt.plot(dictionary2[:,987375],c='b')
# %% grp7 -> short line
plt.plot(dictionary2[:,11072],c='b')
# %% grp8 -> 2 short lines
plt.plot(dictionary2[:,7596],c='b')
plt.plot(dictionary2[:,7462],c='r')
# %% grp9 -> short lines 
plt.plot(dictionary2[:,2055],c='b')
plt.plot(dictionary2[:,2070],c='b')
plt.plot(dictionary2[:,2072],c='b')
plt.plot(dictionary2[:,2075],c='b')

plt.plot(dictionary2[:,2020],c='r')
plt.plot(dictionary2[:,2025],c='r')
plt.plot(dictionary2[:,2029],c='r')
plt.plot(dictionary2[:,2039],c='r')

plt.plot(dictionary2[:,2001],c='g')
plt.plot(dictionary2[:,2008],c='g')

plt.plot(dictionary2[:,2179],c='k')
# %% grp10 -> wider short line
plt.plot(dictionary2[:,35926],c='b')
# %% grp11 -> short lines
plt.plot(dictionary2[:,3280],c='b')
plt.plot(dictionary2[:,3294],c='b')

plt.plot(dictionary2[:,3237],c='r')
plt.plot(dictionary2[:,3244],c='r')

plt.plot(dictionary2[:,3202],c='g')
plt.plot(dictionary2[:,3207],c='g')

plt.plot(dictionary2[:,3173],c='k')
plt.plot(dictionary2[:,3186],c='k')

# %% grp12 ->slightly wider short line
plt.plot(dictionary2[:,20802],c='b')
# %% grp13 -> very thin lines
plt.plot(dictionary2[:,4912],c='b')
plt.plot(dictionary2[:,4919],c='b')

plt.plot(dictionary2[:,4983],c='r')
# %% grp14 -> similar to 20
plt.plot(dictionary2[:,30628],c='b')
# %% grp15 -> very thin
plt.plot(dictionary2[:,6048],c='b')

plt.plot(dictionary2[:,5925],c='r')
# %% grp16 -> similar to 20
plt.plot(dictionary2[:,17242],c='b')
# %% grp17 -> short lines
plt.plot(dictionary2[:,114],c='b')
plt.plot(dictionary2[:,122],c='b')
plt.plot(dictionary2[:,126],c='b')
plt.plot(dictionary2[:,128],c='b')
plt.plot(dictionary2[:,136],c='b')

plt.plot(dictionary2[:,60],c='r')
plt.plot(dictionary2[:,81],c='r')
plt.plot(dictionary2[:,98],c='r')
plt.plot(dictionary2[:,101],c='r')

plt.plot(dictionary2[:,150],c='g')
plt.plot(dictionary2[:,160],c='g')
plt.plot(dictionary2[:,163],c='g')

plt.plot(dictionary2[:,191],c='k')
plt.plot(dictionary2[:,195],c='k')

# %% grp18 -> very thin
plt.plot(dictionary2[:,9186],c='b')

# %% grp19 -> short lines
plt.plot(dictionary2[:,3930],c='b')
plt.plot(dictionary2[:,3962],c='b')

plt.plot(dictionary2[:,3858],c='r')
plt.plot(dictionary2[:,3883],c='r')

plt.plot(dictionary2[:,3841],c='g')
plt.plot(dictionary2[:,3845],c='g')

plt.plot(dictionary2[:,3801],c='k')

# %% grp20 -> similar to 16,14,12,10,5
plt.plot(dictionary2[:,18844],c='b')

# %% grp21 -> thin lines
plt.plot(dictionary2[:,6681],c='b')
plt.plot(dictionary2[:,6695],c='b')
# %% grp22 -> l=imilar to 20
plt.plot(dictionary2[:,12327],c='b')
# %% grp23 -> 2 thin
plt.plot(dictionary2[:,8407],c='b')

plt.plot(dictionary2[:,8260],c='r')
# %% grp24 -> a lot of short lines
plt.plot(dictionary2[:,335],c='b')
plt.plot(dictionary2[:,340],c='b')
plt.plot(dictionary2[:,350],c='b')
plt.plot(dictionary2[:,364],c='b')

plt.plot(dictionary2[:,288],c='b')
plt.plot(dictionary2[:,300],c='b')
plt.plot(dictionary2[:,306],c='b')
plt.plot(dictionary2[:,308],c='b')

plt.plot(dictionary2[:,318],c='b')
plt.plot(dictionary2[:,324],c='b')
plt.plot(dictionary2[:,328],c='b')

plt.plot(dictionary2[:,259],c='b')
plt.plot(dictionary2[:,264],c='b')
plt.plot(dictionary2[:,268],c='b')

plt.plot(dictionary2[:,227],c='b')
plt.plot(dictionary2[:,239],c='b')
plt.plot(dictionary2[:,240],c='b')

plt.plot(dictionary2[:,246],c='b')
plt.plot(dictionary2[:,248],c='b')

plt.plot(dictionary2[:,211],c='k')
# %% grp25 -> short lines
plt.plot(dictionary2[:,4354],c='b')
plt.plot(dictionary2[:,4381],c='b')

plt.plot(dictionary2[:,4319],c='r')
plt.plot(dictionary2[:,4325],c='r')

plt.plot(dictionary2[:,4273],c='g')
plt.plot(dictionary2[:,4289],c='g')

plt.plot(dictionary2[:,4228],c='k')
plt.plot(dictionary2[:,4241],c='k')

# %% grp26 -> short lines
plt.plot(dictionary2[:,897],c='b')
plt.plot(dictionary2[:,911],c='b')
plt.plot(dictionary2[:,915],c='b')
plt.plot(dictionary2[:,921],c='b')

plt.plot(dictionary2[:,982],c='r')
plt.plot(dictionary2[:,988],c='r')
plt.plot(dictionary2[:,996],c='r')

plt.plot(dictionary2[:,958],c='g')
plt.plot(dictionary2[:,966],c='g')

plt.plot(dictionary2[:,939],c='k')
plt.plot(dictionary2[:,941],c='k')

plt.plot(dictionary2[:,866],c='c')
plt.plot(dictionary2[:,879],c='c')
# %% grp27 -> short lines
plt.plot(dictionary2[:,3419],c='b')
plt.plot(dictionary2[:,3436],c='b')
plt.plot(dictionary2[:,3444],c='b')

plt.plot(dictionary2[:,3475],c='r')
plt.plot(dictionary2[:,3493],c='r')

plt.plot(dictionary2[:,3376],c='g')
plt.plot(dictionary2[:,3385],c='g')

plt.plot(dictionary2[:,3344],c='k')
plt.plot(dictionary2[:,3358],c='k')

# %% grp28 -> thin lines
plt.plot(dictionary2[:,8648],c='b')
plt.plot(dictionary2[:,8674],c='b')
# %% grp29 -> thin lines
plt.plot(dictionary2[:,5562],c='b')
plt.plot(dictionary2[:,5586],c='b')

plt.plot(dictionary2[:,5705],c='b')

plt.plot(dictionary2[:,5527],c='b')
# %% grp30 -> short lines
plt.plot(dictionary2[:,392],c='b')
plt.plot(dictionary2[:,398],c='b')
plt.plot(dictionary2[:,401],c='b')
plt.plot(dictionary2[:,406],c='b')
plt.plot(dictionary2[:,415],c='b')
plt.plot(dictionary2[:,432],c='b')

plt.plot(dictionary2[:,462],c='r')
plt.plot(dictionary2[:,478],c='r')
plt.plot(dictionary2[:,481],c='r')
plt.plot(dictionary2[:,488],c='r')

plt.plot(dictionary2[:,500],c='g')
plt.plot(dictionary2[:,512],c='g')
plt.plot(dictionary2[:,523],c='g')
# %% grp31 -> L
plt.plot(dictionary2[:,3003],c='b')
plt.plot(dictionary2[:,3010],c='b')
plt.plot(dictionary2[:,3015],c='b')

plt.plot(dictionary2[:,3036],c='r')
plt.plot(dictionary2[:,3057],c='r')
# %% grp32 -> short lines
plt.plot(dictionary2[:,4579],c='b')
plt.plot(dictionary2[:,4611],c='b')
plt.plot(dictionary2[:,4621],c='b')
plt.plot(dictionary2[:,4637],c='b')

plt.plot(dictionary2[:,4659],c='r')
plt.plot(dictionary2[:,4663],c='r')
plt.plot(dictionary2[:,4669],c='r')

plt.plot(dictionary2[:,4700],c='g')
# %% grp33 -> thin lines
plt.plot(dictionary2[:,5219],c='b')

plt.plot(dictionary2[:,5167],c='r')

plt.plot(dictionary2[:,5108],c='g')
# %% grp34 ->  thin line
plt.plot(dictionary2[:,6308],c='b')

# %% grp35 -> short lines
plt.plot(dictionary2[:,4146],c='b')
plt.plot(dictionary2[:,4179],c='b')

plt.plot(dictionary2[:,4133],c='r')
plt.plot(dictionary2[:,4137],c='r')

plt.plot(dictionary2[:,4096],c='g')

plt.plot(dictionary2[:,4025],c='k')
# %% grp36 -> thin line
plt.plot(dictionary2[:,7280],c='b')
# %% grp37 -> wider spread short lines
plt.plot(dictionary2[:,3666],c='b')
plt.plot(dictionary2[:,3690],c='b')
plt.plot(dictionary2[:,3710],c='b')
plt.plot(dictionary2[:,3735],c='b')

plt.plot(dictionary2[:,3570],c='r')
plt.plot(dictionary2[:,3757],c='r')

plt.plot(dictionary2[:,3607],c='g')

plt.plot(dictionary2[:,3544],c='k')

# %% grp38 -> short lines
plt.plot(dictionary2[:,848],c='b')
plt.plot(dictionary2[:,854],c='b')
plt.plot(dictionary2[:,864],c='b')

plt.plot(dictionary2[:,794],c='r')
plt.plot(dictionary2[:,808],c='r')
plt.plot(dictionary2[:,817],c='r')

plt.plot(dictionary2[:,763],c='g')
plt.plot(dictionary2[:,768],c='g')
plt.plot(dictionary2[:,775],c='g')

plt.plot(dictionary2[:,829],c='k')
plt.plot(dictionary2[:,836],c='k')

plt.plot(dictionary2[:,745],c='c')
plt.plot(dictionary2[:,750],c='c')
# %% grp39 -> thin lines
plt.plot(dictionary2[:,6935],c='b')

plt.plot(dictionary2[:,6866],c='r')
# %% grp40 -> similar to 20
plt.plot(dictionary2[:,17439],c='b')
# %% grp41 -> short lines (thin space)
plt.plot(dictionary2[:,4787],c='b')
plt.plot(dictionary2[:,4798],c='b')
plt.plot(dictionary2[:,4806],c='b')
plt.plot(dictionary2[:,4818],c='b')

plt.plot(dictionary2[:,4751],c='r')
plt.plot(dictionary2[:,4760],c='r')
plt.plot(dictionary2[:,4769],c='r')
# %% grp42 -> short lines (thin space)
plt.plot(dictionary2[:,595],c='b')
plt.plot(dictionary2[:,604],c='b')
plt.plot(dictionary2[:,610],c='b')
plt.plot(dictionary2[:,616],c='b')

plt.plot(dictionary2[:,572],c='r')
plt.plot(dictionary2[:,582],c='r')

plt.plot(dictionary2[:,540],c='g')
plt.plot(dictionary2[:,548],c='g')


# %% 
x2 = np.loadtxt('/Users/komranghahremani/Desktop/channel_2.dat', usecols=(0,1),max_rows=1000)
x3 = np.loadtxt('/Users/komranghahremani/Desktop/channel_3.dat', usecols=(0,1),max_rows=1000)
x4 = np.loadtxt('/Users/komranghahremani/Desktop/channel_4.dat', usecols=(0,1),max_rows=1000)
x5 = np.loadtxt('/Users/komranghahremani/Desktop/channel_5.dat', usecols=(0,1),max_rows=1000)
# %%
plt.plot(x2[600:800,1])
plt.title('Kettle_actual')
#%%
plt.plot(dictionary2[900:,35926])
plt.title('Kettle predicted boxcar')
# %%
14,12,10,5
# %% grp37 -> wider spread short lines
plt.plot(dictionary2[:,3666],c='b')
plt.plot(dictionary2[:,3690],c='b')
plt.plot(dictionary2[:,3710],c='b')
plt.plot(dictionary2[:,3735],c='b')

plt.plot(dictionary2[:,3570],c='r')
plt.plot(dictionary2[:,3757],c='r')

plt.plot(dictionary2[:,3607],c='g')

plt.plot(dictionary2[:,3544],c='k')

#grp35 -> short lines
plt.plot(dictionary2[:,4146],c='b')
plt.plot(dictionary2[:,4179],c='b')

plt.plot(dictionary2[:,4133],c='r')
plt.plot(dictionary2[:,4137],c='r')

plt.plot(dictionary2[:,4096],c='g')

plt.plot(dictionary2[:,4025],c='k')

#grp30 -> short lines
plt.plot(dictionary2[:,392],c='b')
plt.plot(dictionary2[:,398],c='b')
plt.plot(dictionary2[:,401],c='b')
plt.plot(dictionary2[:,406],c='b')
plt.plot(dictionary2[:,415],c='b')
plt.plot(dictionary2[:,432],c='b')

plt.plot(dictionary2[:,462],c='r')
plt.plot(dictionary2[:,478],c='r')
plt.plot(dictionary2[:,481],c='r')
plt.plot(dictionary2[:,488],c='r')

plt.plot(dictionary2[:,500],c='g')
plt.plot(dictionary2[:,512],c='g')
plt.plot(dictionary2[:,523],c='g')

#grp23 -> 2 thin
plt.plot(dictionary2[:,8407],c='b')

plt.plot(dictionary2[:,8260],c='r')

#grp28 -> thin lines
plt.plot(dictionary2[:,8648],c='b')
plt.plot(dictionary2[:,8674],c='b')

# grp4
plt.plot(dictionary2[:,720],c='r')
plt.plot(dictionary2[:,732],c='r')
plt.plot(dictionary2[:,739],c='r')

plt.plot(dictionary2[:,680],c='b')
plt.plot(dictionary2[:,684],c='b')
plt.plot(dictionary2[:,687],c='b')

plt.plot(dictionary2[:,652],c='g')
plt.plot(dictionary2[:,655],c='g')
plt.plot(dictionary2[:,662],c='g')

plt.plot(dictionary2[:,693],c='c')
plt.plot(dictionary2[:,659],c='c')

plt.plot(dictionary2[:,646],c='y')
plt.plot(dictionary2[:,648],c='y')

plt.plot(dictionary2[:,625],c='k')
plt.plot(dictionary2[:,633],c='k')
