import numpy as np
import pickle
import os.path

# TODO Ouput handling


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


def nbatoms_full_set(resultDic):
    """Returns the entire set of number of atom on each sequence k (resultDic[k])."""
    ll_nbatoms = []
    for i in list(resultDic.keys()):
        ll_nbatoms.append(resultDic[i].nbatoms)

    return ll_nbatoms

# ----------------------------------------------------------------------------------------------- #


def Com2resultDic(resultDic, partition, Graph):
    """Store the community detection output in the resultDic.
    """

    # Inveverse the mapping
    dictlistcom = {}
    for k, v in partition.items():
        dictlistcom[v] = []
    for k, v in partition.items():
        dictlistcom[v].append(k)

    # Store the community label for each snippet
    ll_communities = list(dictlistcom.keys())

    for k in resultDic.keys():
        for com in ll_communities:
            ll_p = [int(ppp) for ppp in dictlistcom[com]]
            ind4Kcoef = [i for i, ll in enumerate(
                resultDic[k].REFIND.tolist()) if int(ll) in ll_p]
            resultDic[k].ComDict[com] = resultDic[k].Kcoef[:, ind4Kcoef]

    # Store partition and graph seperately
    resultDic[0].partition = partition
    resultDic[0].Graph = Graph

# ----------------------------------------------------------------------------------------------- #

# TODO find a solution for the saving path

def loadresultDic(name_resultDic, path2store):
    """Load the resultDic.
    """
    # if os.path.exists("/Users/pwin"):
    #     path2store = '/Users/pwin/Documents/nilm_paper/nilm/_store/'

    # else:
    #     path2store = '/work3/s160157/'

    with open(path2store + name_resultDic, 'rb') as handle:
        resultDic = pickle.load(handle)

    return resultDic


def saveresultDic(name_resultDic, resultDic, path2store):
    """Save the resultDic.
    """
    # if os.path.exists("/Users/pwin"):
    #     path2store = '/Users/pwin/Documents/nilm_paper/nilm/_store/'

    # else:
    #     path2store = '/work3/s160157/'

    with open(path2store + name_resultDic, 'wb') as handle:
        pickle.dump(resultDic, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ----------------------------------------------------------------------------------------------- #
