"""Thanks to Sidney Mau https://github.com/sidneymau for kindly providing his code"""
import numpy as np


def raw_moment(arr, p_x, p_y):
    """
    Compute raw moments of arr
    """
    yy, xx = np.indices(arr.shape)

    return np.sum(np.power(xx, p_x) * np.power(yy, p_y) * arr)


def central_moment(arr, p_x, p_y):
    """
    Compute central moments of arr
    """
    yy, xx = np.indices(arr.shape)

    mu_0 = raw_moment(arr, 0, 0)
    mu_x = raw_moment(arr, 1, 0)
    mu_y = raw_moment(arr, 0, 1)
    return np.sum(np.power(xx - mu_x / mu_0, p_x) * np.power(yy - mu_y / mu_0, p_y) * arr)


def moment(arr, p_x, p_y):
    """
    Compute normalized central moments of arr
    """
    return central_moment(arr, p_x, p_y) / central_moment(arr, 0, 0)


class Moments:
    """
    Compute the moments of an array in pixel space
    """

    def __init__(self, arr):
        self.arr = arr

        # Raw moments
        self.R0 = raw_moment(self.arr, 0, 0)
        self.Rx = raw_moment(self.arr, 1, 0)
        self.Ry = raw_moment(self.arr, 0, 1)
        self.Rxx = raw_moment(self.arr, 2, 0)
        self.Ryy = raw_moment(self.arr, 0, 2)
        self.Rxy = raw_moment(self.arr, 1, 1)

        # Central moments
        self.C0 = self.R0
        self.Cx = 0.
        self.Cy = 0.
        self.Cxx = central_moment(self.arr, 2, 0)
        self.Cyy = central_moment(self.arr, 0, 2)
        self.Cxy = central_moment(self.arr, 1, 1)

        # Normalized central moments
        self.M0 = 1.
        self.Mx = 0.
        self.My = 0.
        self.Mxx = moment(self.arr, 2, 0)
        self.Myy = moment(self.arr, 0, 2)
        self.Mxy = moment(self.arr, 1, 1)