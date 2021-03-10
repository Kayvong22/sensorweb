import numpy as np
import pandas as pd
from tools.handling import *

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
