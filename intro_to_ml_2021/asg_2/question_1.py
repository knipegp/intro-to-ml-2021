"""Code for answers to question 1"""
# /usr/bin/env python3

from typing import List

import numpy
import pandas
from intro_to_ml_2021.asg_2.hw2q1 import hw2q1
from numpy import linalg
from sklearn import metrics, preprocessing


def to_frame() -> pandas.DataFrame:
    """Retrieve data samples and convert to dataframe"""
    xtrain, ytrain, xval, yval = hw2q1()
    train = numpy.hstack(
        (
            numpy.array(xtrain).transpose(),
            numpy.array([ytrain]).transpose(),
        )
    )
    val = numpy.hstack(
        (
            numpy.array(xval).transpose(),
            numpy.array([yval]).transpose(),
        )
    )
    frame = pandas.DataFrame(numpy.vstack((train, val)), columns=["x_1", "x_2", "y"])
    types = numpy.vstack(([["train"]] * train.shape[0], [["validate"]] * val.shape[0]))
    frame["type"] = types
    return frame


def simple_design_mat(xvecs: numpy.ndarray) -> numpy.ndarray:
    """Use sklearn to generate design matrix"""
    feats = preprocessing.PolynomialFeatures(degree=3)
    design_mat: numpy.ndarray = feats.fit_transform(xvecs)
    return design_mat


def map_estimate(
    design_mat: numpy.ndarray, yvals: numpy.ndarray, reg: float = 0.0
) -> numpy.ndarray:
    """Do maximum posterior estimation on design matrix"""
    first: numpy.ndarray = numpy.matmul(design_mat.transpose(), design_mat)
    regged = (1 / reg) * numpy.identity(first.shape[0]) + first
    inv = linalg.inv(regged)
    sec: numpy.ndarray = numpy.matmul(inv, design_mat.transpose())
    param_ml: numpy.ndarray = numpy.matmul(sec, numpy.array([yvals]).transpose())
    return numpy.array(param_ml.transpose()[0])


def ml_estimate(
    design_mat: numpy.ndarray,
    yvals: numpy.ndarray,
) -> numpy.ndarray:
    """Do maximum likelihood estimation on the design matrix"""
    first: numpy.ndarray = numpy.matmul(design_mat.transpose(), design_mat)
    inv = linalg.inv(first)
    sec: numpy.ndarray = numpy.matmul(inv, design_mat.transpose())
    param_ml: numpy.ndarray = numpy.matmul(sec, numpy.array([yvals]).transpose())
    return numpy.array(param_ml.transpose()[0])


def lin_estimate(xvals: numpy.ndarray, weights: numpy.ndarray) -> float:
    """Estimate the target for a cubic function"""
    inputs: numpy.ndarray = numpy.array(
        [
            1,
            xvals[0],
            xvals[1],
            pow(xvals[0], 2),
            xvals[0] * xvals[1],
            pow(xvals[1], 2),
            pow(xvals[0], 3),
            pow(xvals[0], 2) * xvals[1],
            xvals[0] * pow(xvals[1], 2),
            pow(xvals[1], 3),
        ]
    )
    target: float = numpy.dot(inputs, weights)
    return target


def get_validate_error(
    weights: numpy.ndarray, xval: numpy.ndarray, yval: numpy.ndarray
) -> float:
    """Get the mean squared errors"""
    guesses: List[float] = [0.0] * len(xval)
    for idx, xvec in enumerate(xval):
        guesses[idx] = lin_estimate(xvec, weights)
    return float(metrics.mean_squared_error(yval, guesses))
