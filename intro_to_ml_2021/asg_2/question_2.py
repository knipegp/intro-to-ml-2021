"""Support for running question 2"""
#!/usr/bin/env python3
from typing import Dict, List, Union
import math

import numpy
import pandas
from intro_to_ml_2021.asg_2 import hw2q1
from numpy import random


def assign_sets(frame: pandas.DataFrame, sets: Dict[str, int]) -> pandas.DataFrame:
    """Randomly assign all samples to a set"""
    sample_cnt: int = sum(list(sets.values()))
    rand_idxs = random.choice(range(sample_cnt), size=sample_cnt, replace=False)
    prev = 0
    for set_name, cnt in sets.items():
        end = cnt + prev
        frame.loc[rand_idxs[prev:end], "set"] = set_name
        prev = end
    return frame


def get_data(
    priors: List[float], means: numpy.ndarray, covs: numpy.ndarray, sets: Dict[str, int]
) -> pandas.DataFrame:
    """Generate the data set for question 2"""
    gmm: Dict[str, Union[List[float], numpy.ndarray]] = {
        "priors": priors,
        "meanVectors": means.transpose(),
        "covMatrices": covs.transpose((1, 2, 0)),
    }
    sample_cnt: int = sum(list(sets.values()))
    raw_samps, raw_labels = hw2q1.generateDataFromGMM(sample_cnt, gmm)
    rltrans = raw_labels[0] - 1
    correct_0 = numpy.where(rltrans == 1, 0, rltrans)
    correct_1 = numpy.where(correct_0 == 2, 1, correct_0)
    labels = numpy.array([correct_1]).transpose()
    samps = raw_samps.transpose()
    frame = pandas.DataFrame(
        numpy.hstack((samps, labels)), columns=["x_1", "x_2", "label"]
    )
    return assign_sets(frame, sets)


def sigmoid_post(weights: numpy.ndarray, phis: numpy.ndarray) -> numpy.ndarray:
    """Calculate guesses for the sigmoid function"""
    var_a = -numpy.matmul(weights, phis)
    guess = 1 / (1 + numpy.exp(var_a))
    return guess


def sig_disc(weights: numpy.ndarray, phis: numpy.ndarray) -> numpy.ndarray:
    """Produce the class guess based on the sigmoid output"""
    guesses = sigmoid_post(weights, phis)
    disc = numpy.where(guesses < 0.5, 0.0, 1.0)
    return disc


def gradient_error(
    param_guess: numpy.ndarray, phis: numpy.ndarray, labels: numpy.ndarray
) -> numpy.ndarray:
    """Calculate the gradient error for the param_guess"""
    guesses = sigmoid_post(param_guess, phis)
    term_2 = guesses - labels
    all_errors = term_2 * phis
    return numpy.sum(all_errors, axis=1)


def get_error(
    weights: numpy.ndarray,
    xvecs: numpy.ndarray,
    labels: numpy.ndarray,
    phis: numpy.ndarray,
):
    """Classify validation"""
    guesses = sig_disc(weights, phis)
    is_correct = guesses == labels
    correct_cnt = numpy.where(is_correct)[0].shape[0]
    return correct_cnt / xvecs.shape[0]


def get_weights(
    labels: numpy.ndarray,
    phis: numpy.ndarray,
    learn_rate: float = 0.0001,
    max_iter: int = 10000,
) -> numpy.ndarray:
    """Run gradient descent to get the model weights"""
    weights = numpy.array([0.0] * phis.shape[0])
    for _ in range(max_iter):
        error = learn_rate * gradient_error(weights, phis, labels)
        new_weights = weights - error
        if numpy.allclose(new_weights, weights):
            break
        weights = new_weights
    return weights
