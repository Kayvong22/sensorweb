import pandas as pd
import numpy as np
from string import punctuation
from collections import Counter
# from funcNavH5 import NavH5


class transformation():
    # TODO : explanation of the function
    """docstring for transformation.
    """

    # Parameters for sequence_timeseries methods:
    resample_rate = 'T'  # alternative '5T'
    interpolation = True
    tol_gap_min = 10
    lower_limit_size_seq = 200
    datalength_degradation_coef = 0.40

    def __init__(self, dictdf):
        """Explanation on the instance variable (self.):

        Args:
            -- self.dictdf: initial instance variable to the class. Usually, nested
            dataframes in a dictionary, cDict_tot & cDict_spec;
            -- self.df: empty instance variable, output of '_2timeseries' methods;
            -- self.seqdict: empty instance variable, output of 'seqDict' methods;
            -- self.tsmonth: empty instance variable, output of 'merge_timeseries'
            methods;
            -- self.tsconcat: empty instance variable, ouput of 'topKappl' method.
        """
        self.dictdf = dictdf
        self.df = None
        self.seqdict = None
        self.tsmonth = None
        self.tsconcat = None
        self.llist = None  # List of the appliances
        self.llappl = None

    @staticmethod
    def interp(df, interptype='bfill'):
        """Compute down (or up) sampling through B-spline method.

        Args:
            -- timestep: as string, e.g. '5min' or 'T'
            -- dftimeseries: Time series as dataframe

        Return the splined measurement as dataframe.
        """
        if interptype == 'bfill':
            # (2) Interpolation through simple filling
            df = df.bfill()

        elif interptype == 'bspline':
            # # Using b-spline of third polynomial order
            df = df.interpolate(method='spline', order=3)

        else:
            print('No interpolation method selected')

        return df

    @staticmethod
    def _2timeseries(df):

        # Correct the negative Active Power
        df['meas'] = df['meas'].abs()

        # Change the indexing to the datetime indexes
        df = df.reset_index(drop=True)

        # Change the format to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Remove the index column to date
        df = df.set_index('date')

        return df

    def merge_phases(self):

        try:
            list(self.dictdf.keys()).index('ActivePowerTotal')

        except ValueError:
            print('No Total Active Power in this dictionary')
            pass

        Dict_ActiveTotPhases = {}

        ll_total_phases = list(self.dictdf.keys())
        iindx = [i for i, ll in enumerate(
            ll_total_phases) if not ll == 'ActivePowerTotal']

        for k in range(len(iindx)):
            self.df = self._2timeseries(self.dictdf[ll_total_phases[iindx[k]]])
            self.sequence_timeseries()  # self.seqdict
            self.merge_timeseries()  # self.tsmonth

            Dict_ActiveTotPhases['Active{}'.format(k + 1)] = self.tsmonth

        self.tsconcat = Dict_ActiveTotPhases['Active1']  # initialise
        self.tsconcat.columns = ['Active1']
        # Eliminate duplicates in the indexes
        self.tsconcat = self.tsconcat[~self.tsconcat.index.duplicated(
            keep='first')]

        for l in range(len(list(Dict_ActiveTotPhases.keys())) - 1):
            tsSPEC = Dict_ActiveTotPhases['Active{}'.format(l + 2)]
            tsSPEC.columns = ['Active{}'.format(l + 2)]
            tsSPEC = tsSPEC[~tsSPEC.index.duplicated(keep='first')]
            self.tsconcat = pd.concat(
                [self.tsconcat, tsSPEC], axis=1, join='inner')

        # Summing the appliances row wise for comparison with given total active power
        self.tsconcat['TotalActivePower_MergePhases'] = self.tsconcat.sum(
            axis=1)

    def _2timeseriesTOT(self):
        """Compute the input dictionary to a pandas time series.
        """
        # ## Seek for Active power total
        indexTotAct = list(self.dictdf.keys()).index('ActivePowerTotal')

        # ## Working on the HDFStore ouput file
        self.df = self.dictdf[list(self.dictdf.keys())[indexTotAct]]
        # Correct the negative Active Power
        self.df['meas'] = self.df['meas'].abs()

        # Change the indexing to the datetime indexes
        self.df = self.df.reset_index(drop=True)

        # Change the format to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Remove the index column to date
        self.df = self.df.set_index('date')

        # return self.df

    def _2timeseriesSPEC(self, str_appl):
        """Compute the input dictionary to a pandas time series.
        """
        self.df = self.dictdf[str_appl]
        self.df['meas'] = self.df['meas'].abs()
        # Change the indexing to the datetime indexes
        self.df = self.df.reset_index(drop=True)
        # Change the format to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        # Remove the index column to date
        self.df = self.df.set_index('date')

    def sequence_timeseries(self):
        """Avoid large gap in the pandas time series by creating
        sequences.

        ** Args:
            --  Defaut interpolation method is 'bfill'. This allows
                the least biasing of the data.
            --  Defaut value is an allowance of 10 minutes maximum,
                above the 10 minutes the time serie in sequenced.
            --  Defaut value is an allowance of 100 minutes for the creation
                of a sequence.

        ** Returns:
            Sequenced data as pandas dataframe nested in a dictionary.
            Overview of the sequence names with "sqdict.keys()"
        """
        # (1) Outliers obtained through delta time threshold
        # Seeking for hole in the dataset, meaning large time difference
        # diff = np.diff(self.df['date'])
        diff = np.diff(self.df.index)
        diff = diff / 1e9  # from nanoseconds to seconds ...

        # Threshold for 10 minutes
        diff_threshold = np.array(
            self.tol_gap_min * 60, dtype='timedelta64[ns]')

        # Outliers check
        outliers = diff > diff_threshold

        # + 1 in the beginning
        indhead = np.where(outliers)[0]
        indhead = np.insert(indhead, 0, 0)

        # + 1 in the end
        indtail = np.where(outliers)[0]
        indtail = np.insert(indtail, len(indtail), len(self.df) - 1)

        # # Graphically check the missing value (or what we considered as outliers)
        self.df = self.df[:-1]  # taking out the last caused by diff()

        self.seqdict = {}
        j = 0
        for i in range(len(indtail)):

            if (indtail[i] - indhead[i]) < self.lower_limit_size_seq:
                continue

            j += 1
            head = indhead[i]
            tail = indtail[i]
            df1 = self.df[head:tail]

            # Resample by Up-sampling and Down-sampling ##########
            df1 = df1.resample(self.resample_rate).mean()

            if self.interpolation == True:
                df1 = self.interp(df=df1)
                self.seqdict[''.join(['seq', str(j)])] = df1

            else:
                self.seqdict[''.join(['seq', str(j)])] = df1

        # return self.seqdict

    def merge_timeseries(self):
        """Merges the sequenced time series from seqDict()
        in one time serie.
        """
        if not bool(self.seqdict):
            self.tsmonth = []

        else:
            self.llist = list(self.seqdict.keys())
            tsframes = [self.seqdict[list] for list in self.llist]

            self.tsmonth = pd.concat(tsframes)
            self.tsmonth = self.tsmonth.sort_index()

    def merge_appliances(self, k_top_appliances=100):
        """Selects the top-k appliances, meaning the k number of appliances
        with the largest energy use.

        The method works in two phase: (1) first phase creates individual
        pandas time series from the appliances (sequencing and merging);
        (2) second phase concatenate the individual time series in one pandas
        dataframe with the aggregated active power.

        ** Args:
            --  k_top_appliances: selection of the top-k appliances.
            --  datalength_degradation_coef: limitation on the degradation of
                the length of the dataset. Defaut value = 0.40;

        ** Returns:
            --  dataframe with the k-appliances column wise and their
                aggregated active power.
        """
        self.llappl = list(self.dictdf.keys())

        Dict_tsmonth = {}
        ll_length_ts = []

        # (1) ###
        for i, ll in enumerate(self.llappl):

            self._2timeseriesSPEC(str_appl=self.llappl[i])
            self.sequence_timeseries()  # self.seqdict
            self.merge_timeseries()  # self.tsmonth

            # Maybe in merge_timeseries methods ... but then self.i necessary
            self.tsmonth.columns = [self.llappl[i]]
            Dict_tsmonth['tsSPEC{}'.format(i + 1)] = self.tsmonth
            ll_length_ts.append('tsSPEC{}'.format(i + 1))
            ll_length_ts.append(len(self.tsmonth))

        matrixll = np.array(ll_length_ts)
        shape = (int(len(ll_length_ts) / 2), 2)
        output_ll_length_ts = matrixll.reshape(shape)

        df_length = pd.DataFrame(output_ll_length_ts)
        df_length.columns = ['varname', 'length']
        df_length['length'] = df_length['length'].astype(int)

        df_length = df_length.sort_values(by=['length'], ascending=False)

        # (2) ### NEW ###
        # Retrieve the variable with the longest ts
        self.tsconcat = Dict_tsmonth[df_length.iloc[0, 0]]
        # Eliminate duplicates in the indexes
        self.tsconcat = self.tsconcat[~self.tsconcat.index.duplicated(
            keep='first')]

        for l in range(len(list(Dict_tsmonth.keys())) - 1):
            tsSPEC = Dict_tsmonth[df_length.iloc[l + 1, 0]]
            tsSPEC = tsSPEC[~tsSPEC.index.duplicated(keep='first')]

            lendiff = len(tsSPEC) / len(self.tsconcat)

            if lendiff > self.datalength_degradation_coef:
                self.tsconcat = pd.concat(
                    [self.tsconcat, tsSPEC], axis=1, join='inner')

            else:
                break
        # NEW ##

        # Merging the multiphase appliances
        self.merge_appliance_phases()  # changes the tsconcat in the "background"

        # (3) - Preparing the dataframe for output
        # Top-k appliances
        topk = self.tsconcat.sum().nlargest(k_top_appliances) / 60
        iind = list(topk.index)

        # Summing the appliances row wise for comparison with given total active power
        self.tsconcat['TotalPower_Appliances'] = self.tsconcat.loc[:, iind].sum(
            axis=1)

        # return self.tsconcat

    def merge_appliance_phases(self):
        """ Merges the appliances connected on multiple phases.
        """

        # Part regarding the appliances on multiple phases ---------------------------
        ll_colnames = list(self.tsconcat.columns)
        ll_nophase = [ll_colnames[i].split('_')[0]
                      for i in range(len(ll_colnames))]

        name_multiPhase_appl = sorted(
            Counter(ll_nophase) - Counter(set(ll_nophase)))

        # Condition with no appliances on multi phase ...
        if not name_multiPhase_appl:
            print('No multi phase appliances')
            pass

        else:
            print('1 or more appliances on more than one phase')

            # Looping around the differente multi phase appliances ...
            for ll_appl in name_multiPhase_appl:

                ll_colnames = list(self.tsconcat.columns)
                ll_nophase = [ll_colnames[i].split('_')[0]
                              for i in range(len(ll_colnames))]
                # Find the multiple indices
                indices = [i for i, ll in enumerate(
                    ll_nophase) if ll == ll_appl]

                # Merging the multiphase appliances in one and drop the merged columns
                self.tsconcat[ll_appl] = self.tsconcat.iloc[:,
                                                            indices].sum(axis=1)
                self.tsconcat = self.tsconcat.drop(
                    [ll_colnames[iind] for iind in indices], axis=1)

        ll_colnames = list(self.tsconcat.columns)
        ll_nophase = [ll_colnames[i].split('_')[0]
                      for i in range(len(ll_colnames))]
        self.tsconcat.columns = ll_nophase
