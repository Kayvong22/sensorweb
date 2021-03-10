import pandas
import numpy
import pandas as pd
import numpy as np
from tools.rle import rlencode


class snippet4omp(object):
    """Create snippets for the sparse decomposition.

    Input: - signal:
    """

    threshold_small_power = None
    threshold_encoding = None
    threshold_restrict = None

    def __init__(self, signal):

        assert isinstance(
            signal, (pandas.core.series.Series,
                     pandas.core.frame.DataFrame,
                     numpy.ndarray)),   "Wrong input type, signal should be pandas.series OR pandas.df OR numpy.array"

        if isinstance(signal, pandas.core.series.Series):
            signal = signal
            self.signal = signal.values

        if isinstance(signal, pandas.core.frame.DataFrame):
            signal = signal
            self.signal = signal.values


        else:
            self.signal = signal.values
            pass

    def negl_small_power(self, threshold=70):
        """Neglect the small power draw.
        Default: threshold: 70[W]
        """

        self.threshold_small_power = threshold

        ind_below = np.where(self.signal < threshold)[0]
        # ind_above = np.where(self.signal > 70)[0]
        self.signal[ind_below] = 0

    def encoding_start_end(self, threshold=15):
        """Computing the snippet via indices vectors (start and end).

        Args: - threshold: a break creating two snippet should occur when no big power draw during
        XX (== threshold) time sample.
        """

        self.threshold_encoding = threshold

        rle_y_hat = {}
        rle_list = ['starts', 'lenghts', 'values']

        for i, ll in enumerate(rle_list):
            rle_y_hat[ll] = rlencode(self.signal-threshold)[i]

        rle_y_hat['values'][rle_y_hat['values'] > 0] = 1
       
        xxx = (1-rle_y_hat['values'])*rle_y_hat['lenghts']

        start = rle_y_hat['starts']
        end = rle_y_hat['starts'] + rle_y_hat['lenghts']
    
        #at 5min
        #start2=start[np.where(xxx > threshold)[0]] 

        start2=start[np.where(xxx > 240)[0]] 
       
        #self.start_omp_loop=start2
        #at 5min hp detection
        self.start_omp_loop = start2#np.concatenate([[0],np.array([start2[j]for j in [np.min(np.where(np.floor((np.cumsum(start2[1:]-start2[:-1])/100)+0.1)==i)[0]) for i in np.unique(np.floor((np.cumsum(start2[1:]-start2[:-1])/100)+0.1))]])+2])
        
        #self.start_omp_loop = np.concatenate([[0],np.array([start2[j]for j in [np.min(np.where(np.floor((np.cumsum(start2[1:]-start2[:-1])/25)+0.1)==i)[0]) for i in np.unique(np.floor((np.cumsum(start2[1:]-start2[:-1])/25)+0.1))]])+2])
        end_omp_loop = self.start_omp_loop[1:]
        if  end_omp_loop[-1]!=len(self.signal):
            self.end_omp_loop = np.concatenate([end_omp_loop, [len(self.signal)]])
        else:
            self.end_omp_loop =end_omp_loop

    # def encoding_start_end(self, threshold=15):
    #     """Computing the snippet via indices vectors (start and end).
    #     Args: - threshold: a break creating two snippet should occur when no big power draw during
    #     XX (== threshold) time sample.
    #     """

    #     self.threshold_encoding = threshold

    #     rle_y_hat = {}
    #     rle_list = ['starts', 'lenghts', 'values']

    #     for i, ll in enumerate(rle_list):
    #         rle_y_hat[ll] = rlencode(self.signal)[i]

    #     rle_y_hat['values'][rle_y_hat['values'] > 0] = 1
    #     rle_values = rle_y_hat['values']
    #     rle_values = 1 - rle_values

    #     start = rle_y_hat['starts']
    #     end = rle_y_hat['starts'] + rle_y_hat['lenghts']

    #     rle_timechange = np.concatenate([[0], end])
    #     rle_difftchange = rle_timechange[1:] - rle_timechange[:-1]

    #     xxx = rle_difftchange * rle_values

    #     iind = np.where(xxx > threshold)[0]

    #     self.start_omp_loop = start[iind]
    #     end_omp_loop = self.start_omp_loop[1:]
    #     self.end_omp_loop = np.concatenate([end_omp_loop, [len(self.signal)]])

    def restrict_start_end(self, threshold=1000):
        """Restrict the start and end sequence for the OMP according to the dictionary size.

        Indeed, the dictionary generated in omp.dictionary should be the same length as the
        longest sequence.

        Args: - threshold: max length of a input sequence in the omp algorithm.
        """

        self.threshold_restrict = threshold

        iind = np.where(
            (self.end_omp_loop - self.start_omp_loop) > threshold)[0]

        while (len(iind) != 0):
            ts = self.signal[self.start_omp_loop[iind[0]]:self.end_omp_loop[iind[0]]]
            half_length = int(len(ts) / 2)

            restr_ts = ts[half_length:]
            val_min = min(restr_ts)


            ind_val_min = np.where(val_min == restr_ts)[0][0]

            addup = half_length + ind_val_min

            self.start_omp_loop = np.insert(
                self.start_omp_loop, iind[0] + 1, self.start_omp_loop[iind[0]] + addup)
            self.end_omp_loop = np.insert(
                self.end_omp_loop, iind[0], self.start_omp_loop[iind[0]] + addup)

            iind = np.where((self.end_omp_loop - self.start_omp_loop) > threshold)[0]

    def del_snippet_zero_power(self):
        """Deletion of snippet with zero power.
        """

        self.ll_i_omp = []
        for i, (i_omp, j) in zip(self.start_omp_loop, enumerate(self.end_omp_loop)):

            if max(self.signal[i:j]) == float(0.0):
                self.ll_i_omp.append(i_omp)
            else:
                continue

        self.start_omp_loop = np.delete(self.start_omp_loop, self.ll_i_omp)
        # "-1" to create a continuous sequence with the previsou sequence
        self.end_omp_loop = np.delete(
            self.end_omp_loop, list(np.array(self.ll_i_omp) - 1))

    def del_df_zero_power(self, dfappl):
        """Deletion of the corresponding dataframe chunk from "del_snippet_zero_power.
        """

        ll_ind_del = []
        for i in range(len(self.ll_i_omp)):
            a = self.start_omp_loop[self.ll_i_omp[i]]
            b = self.end_omp_loop[self.ll_i_omp[i]]

            ll_ind_del.append(list(np.arange(a, b, 1)))

        ll_ind_del = [l for ll in ll_ind_del for l in ll]  # list flatten

        self.dfappl = dfappl.drop(dfappl.index[[ll_ind_del]])
