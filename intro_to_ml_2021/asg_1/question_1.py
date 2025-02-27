"""General functions for solving question 1"""
# /usr/bin/env python3

from typing import Callable, Dict, List, Tuple

import numpy
from numpy import random
from scipy import stats


def between(high: float, inc_low: float) -> Callable[[float], bool]:
    """Create a function that returns true if the passed value is under high and
    is greater than or equal to low"""
    return lambda x: high > x >= inc_low


def create_label_lims(priors: List[float]) -> Dict[int, Callable[[float], bool]]:
    """Create limits for each class based on prior probabilities"""
    last_high: float = 0.0
    ret: Dict[int, Callable[[float], bool]] = dict()
    for idx, prop in enumerate(priors):
        new_high = prop + last_high
        ret[idx] = between(new_high, last_high)
        last_high = new_high
    return ret


def classify(val: float, classifier: Dict[int, Callable[[float], bool]]) -> int:
    """Find the class the sample belongs to"""
    for label, is_between in classifier.items():
        if is_between(val):
            return label
    raise ValueError(f"value, {val} cannot be classified")


def gen_label_cnt(priors: List[float], total: int) -> List[int]:
    """Generate the number of each label given priors"""
    gen: random.Generator = random.default_rng()
    vals = gen.random(total)
    lims = create_label_lims(priors)

    labelled: List[int] = [0] * len(priors)
    for samp in vals:
        label = classify(samp, lims)
        labelled[label] += 1
    return labelled


def gen_labelled_samples(
    priors: List[float],
    dists: List[
        stats._multivariate.multivariate_normal_frozen  # pylint: disable=protected-access
    ],
    sample_cnt: int,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Generate the question one samples with labels"""
    label_cnts = gen_label_cnt(priors, sample_cnt)
    samples = numpy.array(list([0] * dists[0].dim), dtype=float, ndmin=2)
    labels = numpy.array(list([0]), dtype=int)
    for label, cnt in enumerate(label_cnts):
        samples = numpy.vstack((samples, dists[label].rvs(size=cnt)))
        labels = numpy.append(labels, [label] * cnt)
    return samples[1:], labels[1:]


def get_threshes(cnt: int, ideal: float) -> Tuple[numpy.ndarray, int]:
    """Get a list of all thresholds to test"""
    tmax = ideal * 10
    step = tmax / cnt
    arr = numpy.arange(0, ideal, step)
    ideal_idx = len(arr)
    arr = numpy.append(arr, ideal)
    arr = numpy.concatenate((arr, numpy.arange(ideal, tmax, step)))
    return (arr, ideal_idx)
