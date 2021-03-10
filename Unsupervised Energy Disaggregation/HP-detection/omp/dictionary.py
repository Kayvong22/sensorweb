import numpy as np
import time

def Heaviside(x, a):
    """Compute a Heaviside function."""
    y = (np.sign(x - a) + 1) / 2
    y[y == 0.5] = 0
    return y


def Boxcar(l, w, t):
    """Compute a boxcar function.

    Arguments:
        l -- constant in the interval (non-zero)
        w -- width of the boxcar, i.e. the internval equal to constant 'l'
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
            X2[:, i] = Boxcar(i, j, x)
            # boxcarinfos[j, i] = j, i
        X = np.concatenate((X, X2), axis=1)

    if not infos:
        return X

    else:
        return X, boxcarinfos
