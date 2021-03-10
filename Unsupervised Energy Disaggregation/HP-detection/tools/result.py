import numpy as np

# TODO Verify no superfluous call

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
