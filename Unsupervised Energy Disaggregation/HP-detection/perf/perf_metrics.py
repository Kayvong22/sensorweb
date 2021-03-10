import numpy as np
import pandas as pd
import pickle
import scipy.stats as scistat
import matplotlib.pyplot as plt


# TODO : clean up

# path = '_store/'
# with open(path + 'y_hat_dict.pickle', 'rb') as handle:
#     y_hat_dict = pickle.load(handle)
#
# with open(path + 'y_truth_dict.pickle', 'rb') as handle:
#     y_truth_dict = pickle.load(handle)
#
# y_hat = y_hat_dict['dishwasher']
# y_truth = y_truth_dict['dishwasher']


# --------- GROUND TRUTH AND BIAS --------------- #

def est_acc(y_hat, y_truth):
    """Estimated accuracy."""

    a = (np.sum(np.abs(y_hat - y_truth)))
    b = (2 * np.sum(y_truth))

    return 1 - (a / b)


def mae(y_hat, y_truth):
    """Mean absolute error."""
    return np.abs((y_truth - y_hat)).mean()


def rmse(y_hat, y_truth):
    """Root mean square error."""
    return np.sqrt(((y_hat - y_truth) ** 2).mean())


# est_acc(y_hat, y_truth)
# mae(y_hat, y_truth)
# rmse(y_hat, y_truth)


# --------- CLASSIFICATION ACCURACY --------------- #
# By Kelly and Makonin

def true_positives(y_hat, y_truth):
    """Correctly predicted that the appliance was ON."""
    binary_y_hat = (y_hat > 0)
    binary_y_truth = (y_truth > 0)

    return len(np.where((binary_y_hat == True) & (binary_y_truth == True))[0])


def true_negatives(y_hat, y_truth):
    """Correctly predicted that the appliance was OFF."""
    binary_y_hat = (y_hat > 0)
    binary_y_truth = (y_truth > 0)

    return len(np.where((binary_y_hat == False) & (binary_y_truth == False))[0])


def false_positives(y_hat, y_truth):
    """Predicted appliance was ON but was OFF."""
    binary_y_hat = (y_hat > 0)
    binary_y_truth = (y_truth > 0)

    return len(np.where((binary_y_hat == True) & (binary_y_truth == False))[0])


def false_negatives(y_hat, y_truth):
    """Appliance was ON but was predicted OFF."""
    binary_y_hat = (y_hat > 0)
    binary_y_truth = (y_truth > 0)

    return len(np.where((binary_y_hat == False) & (binary_y_truth == True))[0])


def recall(y_hat, y_truth):
    """Harmonic mean of recall."""
    try:
        r = (true_positives(y_hat, y_truth)) / (true_positives(y_hat, y_truth) + false_negatives(y_hat, y_truth))
    except ZeroDivisionError:
        r = 0

    return r


def precision(y_hat, y_truth):
    """Harmonic mean of precision."""
    try:
        p = (true_positives(y_hat, y_truth)) / (true_positives(y_hat, y_truth) + false_positives(y_hat, y_truth))
    except ZeroDivisionError:
        p = 0

    return p


def positives_ground_truth(y_hat, y_truth):
    """Number of positives in ground truth."""
    binary_y_truth = (y_truth > 0)

    return len(np.where(binary_y_truth == True)[0])


def negatives_ground_truth(y_hat, y_truth):
    """Number of negatives in ground truth."""
    binary_y_truth = (y_truth > 0)

    return len(np.where(binary_y_truth == False)[0])


def f1_score(y_hat, y_truth):
    a = precision(y_hat, y_truth) * recall(y_hat, y_truth)
    b = precision(y_hat, y_truth) + recall(y_hat, y_truth)

    try:
        f1 = 2 * a / b
    except ZeroDivisionError:
        f1 = 0

    return f1


def accuracy(y_hat, y_truth):
    a = true_positives(y_hat, y_truth) + true_negatives(y_hat, y_truth)
    b = positives_ground_truth(y_hat, y_truth) + \
        negatives_ground_truth(y_hat, y_truth)

    try:
        acc = a / b
    except ZeroDivisionError:
        acc = 0

    return acc


# recall(y_hat, y_truth)
# precision(y_hat, y_truth)
# f1_score(y_hat, y_truth)
# accuracy(y_hat, y_truth)


# --------- ENERGY RELATED --------------- #

def proportion_energy_tp(y_hat, y_truth):
    """Proportion of total energy correctly assigned."""

    fx_y_hat = y_hat
    area = np.sum(fx_y_hat) * (len(fx_y_hat) - 1) / len(fx_y_hat)
    energy_hat = area * 1 / 60

    fx_y_truth = y_truth
    area = np.sum(fx_y_truth) * (len(fx_y_truth) - 1) / len(fx_y_truth)
    energy_truth = area * 1 / 60

    a = np.abs(energy_hat - energy_truth)
    b = max(energy_truth, energy_hat)

    return a / b


def relative_err_energy(y_hat, y_truth):
    """Relative error in total energy."""
    pass


# proportion_energy_tp(y_hat, y_truth)
