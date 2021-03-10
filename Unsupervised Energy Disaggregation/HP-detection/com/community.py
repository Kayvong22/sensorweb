import numpy as np
import pandas as pd
import itertools
import networkx as nx
from com.graph import *


class Community(Graph):
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
