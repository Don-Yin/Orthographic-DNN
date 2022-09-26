from __future__ import division

# from utils.analyse.describe_human_data import DescribeHumanData
# DescribeHumanData()
# from utils.data_generate.main import init_create_train_data
# init_create_train_data()
from itertools import product

"""
Functions for calculating the statistical significant differences between two dependent or independent correlation
coefficients.
The Fisher and Steiger method is adopted from the R package http://personality-project.org/r/html/paired.r.html
and is described in detail in the book 'Statistical Methods for Psychology'
The Zou method is adopted from http://seriousstats.wordpress.com/2012/02/05/comparing-correlations/
Credit goes to the authors of above mentioned packages!

Author: Philipp Singer (www.philippsinger.info)
"""


__author__ = "psinger"

from math import atanh, pow

import numpy as np
from numpy import tanh
from scipy.stats import norm, t


def rz_ci(r, n, conf_level=0.95):
    zr_se = pow(1 / (n - 3), 0.5)
    moe = norm.ppf(1 - (1 - conf_level) / float(2)) * zr_se
    zu = atanh(r) + moe
    zl = atanh(r) - moe
    return tanh((zl, zu))


def rho_rxy_rxz(rxy, rxz, ryz):
    num = (ryz - 1 / 2.0 * rxy * rxz) * (1 - pow(rxy, 2) - pow(rxz, 2) - pow(ryz, 2)) + pow(ryz, 3)
    den = (1 - pow(rxy, 2)) * (1 - pow(rxz, 2))
    return num / float(den)


def dependent_corr(xy, xz, yz, n, twotailed=True, conf_level=0.95, method="steiger"):
    """
    Calculates the statistic significance between two dependent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between x and z
    @param yz: correlation coefficient between y and z
    @param n: number of elements in x, y and z
    @param twotailed: whether to calculate a one or two tailed test, only works for 'steiger' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'steiger' or 'zou'
    @return: t and p-val
    """
    if method == "steiger":
        d = xy - xz
        determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
        av = (xy + xz) / 2
        cube = (1 - yz) * (1 - yz) * (1 - yz)

        t2 = d * np.sqrt((n - 1) * (1 + yz) / (((2 * (n - 1) / (n - 3)) * determin + av * av * cube)))
        p = 1 - t.cdf(abs(t2), n - 3)

        if twotailed:
            p *= 2

        return t2, p
    elif method == "zou":
        L1 = rz_ci(xy, n, conf_level=conf_level)[0]
        U1 = rz_ci(xy, n, conf_level=conf_level)[1]
        L2 = rz_ci(xz, n, conf_level=conf_level)[0]
        U2 = rz_ci(xz, n, conf_level=conf_level)[1]
        rho_r12_r13 = rho_rxy_rxz(xy, xz, yz)
        lower = xy - xz - pow((pow((xy - L1), 2) + pow((U2 - xz), 2) - 2 * rho_r12_r13 * (xy - L1) * (U2 - xz)), 0.5)
        upper = xy - xz + pow((pow((U1 - xy), 2) + pow((xz - L2), 2) - 2 * rho_r12_r13 * (U1 - xy) * (xz - L2)), 0.5)
        return lower, upper
    else:
        raise Exception("Wrong method!")


def independent_corr(xy, ab, n, n2=None, twotailed=True, conf_level=0.95, method="fisher"):
    """
    Calculates the statistic significance between two independent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between a and b
    @param n: number of elements in xy
    @param n2: number of elements in ab (if distinct from n)
    @param twotailed: whether to calculate a one or two tailed test, only works for 'fisher' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'fisher' or 'zou'
    @return: z and p-val
    """

    if method == "fisher":
        xy_z = 0.5 * np.log((1 + xy) / (1 - xy))
        ab_z = 0.5 * np.log((1 + ab) / (1 - ab))
        if n2 is None:
            n2 = n

        se_diff_r = np.sqrt(1 / (n - 3) + 1 / (n2 - 3))
        diff = xy_z - ab_z
        z = abs(diff / se_diff_r)
        p = 1 - norm.cdf(z)
        if twotailed:
            p *= 2

        return z, p
    elif method == "zou":
        L1 = rz_ci(xy, n, conf_level=conf_level)[0]
        U1 = rz_ci(xy, n, conf_level=conf_level)[1]
        L2 = rz_ci(ab, n2, conf_level=conf_level)[0]
        U2 = rz_ci(ab, n2, conf_level=conf_level)[1]
        lower = xy - ab - pow((pow((xy - L1), 2) + pow((U2 - ab), 2)), 0.5)
        upper = xy - ab + pow((pow((U1 - xy), 2) + pow((ab - L2), 2)), 0.5)
        return lower, upper
    else:
        raise Exception("Wrong method!")


# print(dependent_corr(0.40, 0.50, 0.10, 103, method="steiger"))
# print(independent_corr(0.5, 0.6, 103, 103, method="fisher"))


# result = independent_corr(0.5, 0.6, 103, 103, method="fisher")

# correlations = [0.5, 0.7, 0.58, 0.71, 0.72, 0.61, 0.65, 0.38, 0.25, 0.36, 0.31, 0.53, 0.56, 0.67, 0.66, 0.44, 0.66, 0.57, 0.7, 0.69]

# combinations = list(product(correlations, correlations))
# combinations = [x for x in combinations if x[0] != x[1]]

# print(combinations)

# for i in combinations:
#     result = independent_corr(i[0], i[1], 28, method="fisher")
#     if result[1] < 0.05:
#         print(i, result)

result = independent_corr(0.6385714285714286, 0.325, 28, method="fisher")
print(result)
# print dependent_corr(.396, .179, .088, 200, method='zou')
# print independent_corr(.560, .588, 100, 353, method='zou')
