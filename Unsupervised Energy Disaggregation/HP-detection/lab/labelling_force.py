import numpy as np

from tools.rle import rlencode
from perf.perf_metrics import *


class labelling_force(object):
    """Labelling process requires so far the time series, outputed from the
    community detection, and the ground truth about the appliances/rooms
    power draw. Later on the ground truth will become a dictionary of past
    events.

    Args:
        -- y_hat: unlabelled appliances channel
        -- y: ground truth appliances channel

    Return:
        -- Labelled channel."""

    def __init__(self, resultDic):
        self.resultDic = resultDic
        self.ll_columns = list(self.resultDic[0].dataframe.drop(
            ['TotalPower_Appliances'], axis=1))

    def build_empty_dict(self):
        """ ."""
        ll_communities = list(self.resultDic[0].ComDict.keys())
        length_df = np.cumsum([len(self.resultDic[i].signal)
                               for i in self.resultDic.keys()])[-1]

        # Community to time series appliances channel
        y_com_dict = {}
        for com in ll_communities:
            y_com_dict[com] = np.concatenate(
                [self.resultDic[i].ComDict[com].sum(axis=1) for i in self.resultDic.keys()])

        # Ground truth appliances channel
        y_truth_dict = {}
        for ll in self.ll_columns:
            y_truth_dict[ll] = np.concatenate(
                [self.resultDic[i].dataframe[ll].values for i in self.resultDic.keys()])

        # Predictited appliances channel
        y_hat_dict = {}
        for ll in self.ll_columns:
            y_hat_dict[ll] = np.zeros(length_df)

        return y_com_dict, y_truth_dict, y_hat_dict

    @staticmethod
    def find_snippets(y_com_indiv, tol_snippet=15):
        """Returns the (start, end) of the snippets."""

        rle_y_hat = {}
        rle_list = ['starts', 'lenghts', 'values']

        for i, ll in enumerate(rle_list):
            rle_y_hat[ll] = rlencode(y_com_indiv)[i]

        # ------------------------------------------------------------------- #
        start = rle_y_hat['starts']
        end = rle_y_hat['starts'] + rle_y_hat['lenghts']

        rle_timechange = np.concatenate([[0], end])
        rle_difftchange = rle_timechange[1:] - rle_timechange[:-1]

        # ------------------------------------------------------------------- #

        rle_y_hat['values'][rle_y_hat['values'] > 0] = 1
        rle_values = rle_y_hat['values']
        rle_values = 1 - rle_values

        xxx = rle_difftchange * rle_values

        iind = np.where(xxx > tol_snippet)[0]

        start_new = start[iind][1:]
        end_new = end[iind][:-1]

        snippets = {}
        for i, (j, k) in zip(start_new, enumerate(end_new)):
            snippets[j] = {}
            snippets[j]['end'] = i
            snippets[j]['start'] = k

        return snippets

    @staticmethod
    def best_label_snippet(y_com_indiv, snippets, y_truth_dict):
        """Assign the best label for a particular time series

        Return:
            -- best label for 1 time series

        ."""
        ll_length = []
        ll_metric = []

        for appl_name in y_truth_dict.keys():
            for i in snippets.keys():

                temp_start = snippets[i]["start"]
                temp_end = snippets[i]["end"]

                a = y_com_indiv[temp_start:temp_end]
                b = y_truth_dict[appl_name][temp_start:temp_end]

                ll_length.append(temp_end - temp_start)
                # ll_metric.append(rmse(y_hat=a, y_truth=b))
                ll_metric.append(est_acc(y_hat=a, y_truth=b))

        row_shape = len(y_truth_dict.keys())
        col_shape = int(len(ll_metric) / row_shape)

        collection_lengths = np.reshape(
            np.array(ll_length), (row_shape, col_shape))
        collection_metric = np.reshape(
            np.array(ll_metric), (row_shape, col_shape))

        ll_weight_metric = []
        for i, appl_name in enumerate(y_truth_dict.keys()):

            tot_length_snippets = np.sum(collection_lengths[i])
            w_metric = np.sum(
                collection_lengths[i] / tot_length_snippets * collection_metric[i])

            ll_weight_metric.append(w_metric)

        ind_max = ll_weight_metric.index(max(ll_weight_metric))
        label4ts = list(y_truth_dict.keys())[ind_max]

        return label4ts

    @staticmethod
    def best_label_no_snippet(y_com_indiv, y_truth_dict):
        """ ."""
        ll_metric = []

        for appl_name in y_truth_dict.keys():
            a = y_com_indiv
            b = y_truth_dict[appl_name]

            ll_metric.append(est_acc(y_hat=a, y_truth=b))

        ind_max = ll_metric.index(max(ll_metric))
        label4ts = list(y_truth_dict.keys())[ind_max]

        return label4ts

    def best_permutation_name(self):
        """Compute the list with the best permutation name.
        """

        y_com_dict, self.y_truth_dict, self.y_hat_dict = self.build_empty_dict()
        ll_communities = list(self.resultDic[0].ComDict.keys())

        ll_best_permutation_name = []
        for com in ll_communities:
            y_com_indiv = y_com_dict[com]

            best_permutation_name = self.best_label_no_snippet(
                y_com_indiv, self.y_truth_dict)
            self.y_hat_dict[best_permutation_name] = self.y_hat_dict[best_permutation_name] + y_com_indiv

            ll_best_permutation_name.append(best_permutation_name)
