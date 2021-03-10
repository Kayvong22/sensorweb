import numpy as np
from tools.rle import rlencode


class snippets4lab(object):
    """docstring for snippets4lab."""

    def __init__(self, timeseries, tol_snippet=25, deletion_zero_snippets=True):
        self.timeseries = timeseries
        self.tol_snippet = tol_snippet
        self.deletion_zero_snippets = deletion_zero_snippets

    def snippet_start_stop(self):
        """Return the starts and ends for the snippet of the sequence. A snippet is
        created when the time series reach for a certain period 0.abs

        Args:
            -- timeseries: signal input
            -- tol_snippet: tolerance to create a snippet. Default value is 25, i.e.
            25 minutes of steadyness to create a snippet.
        """

        ind_below = np.where(self.timeseries < 50)[0]
        ind_above = np.where(self.timeseries > 50)[0]
        self.timeseries[ind_below] = 0

        rle_y_hat = {}
        rle_list = ['starts', 'lenghts', 'values']

        for i, ll in enumerate(rle_list):
            rle_y_hat[ll] = rlencode(self.timeseries)[i]

        rle_y_hat['values'][rle_y_hat['values'] > 0] = 1
        rle_values = rle_y_hat['values']
        rle_values = 1 - rle_values

        start = rle_y_hat['starts']
        end = rle_y_hat['starts'] + rle_y_hat['lenghts']

        rle_timechange = np.concatenate([[0], end])
        rle_difftchange = rle_timechange[1:] - rle_timechange[:-1]

        xxx = rle_difftchange * rle_values

        iind = np.where(xxx > self.tol_snippet)[0]

        self.start_omp_loop = start[iind]
        # self.end_omp_loop = self.start_omp_loop[1:] - 1
        self.end_omp_loop = self.start_omp_loop[1:]
        self.end_omp_loop = np.concatenate(
            [self.end_omp_loop, [len(self.timeseries)]])

        if self.deletion_zero_snippets == True:
            # ll_i_omp = []
            for i, (i_omp, j) in zip(self.start_omp_loop, enumerate(self.end_omp_loop)):
                if max(self.timeseries[i:j]) == float(0.0):
                    # ll_i_omp.append(i_omp)
                    self.start_omp_loop = np.delete(self.start_omp_loop, i_omp)
                    self.end_omp_loop = np.delete(self.end_omp_loop, i_omp - 1)
                else:
                    continue

    def dict_snippets(self):

        self.snippet_start_stop()

        i_omp = 0  # initialise
        ll_i_omp = []
        result = {}

        for i, (i_omp, j) in zip(self.start_omp_loop, enumerate(self.end_omp_loop)):

            ll_i_omp.append(i_omp)
            array_TEMP = self.timeseries[i:j]

            result[i_omp] = array_TEMP

        return result

    def dict_snippets_without_zeros(self):
        # Takes out the zeros arrays

        self.snip_dict = self.dict_snippets()

        new_snip_dict_keys = [
            i for i in self.snip_dict.keys() if max(self.snip_dict[i]) != float(0.0)]

        new_snip_dict = {}
        for i in new_snip_dict_keys:
            new_snip_dict[i] = self.snip_dict[i]

        self.snip_dict = new_snip_dict

        return self.snip_dict
