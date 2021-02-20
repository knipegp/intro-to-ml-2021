"""Functions for question 2"""
#!/usr/bin/env python3
from typing import Callable, List, Optional, Tuple, Dict

from numpy import random
import numpy
import pandas
import seaborn
from matplotlib import axes, pyplot
from scipy import stats


def gen_random_means(
    dim: int,
    mean_range: Tuple[float, float],
    prev: Optional[
        stats._multivariate.multivariate_normal_frozen  # pylint: disable=protected-access
    ],
) -> numpy.ndarray:
    rand_gen = random.default_rng()
    if prev is None:
        use_range: Tuple[float, float] = mean_range
    else:
        avg_std = (numpy.diagonal(prev.cov) ** 0.5).mean()
        new_mean = prev.mean + 2 * avg_std
        use_range: Tuple[float, float] = (
            new_mean + mean_range[0],
            new_mean + mean_range[1],
        )
    return rand_gen.random(dim) * (use_range[1] - use_range[0]) + use_range[0]


def gen_cov_matrix(
    dim: int, std_range: Tuple[float, float], cov_range: Tuple[float, float]
) -> numpy.ndarray:
    rand_gen = random.default_rng()
    stdevs: numpy.ndarray = (
        rand_gen.random(dim) * (std_range[1] - std_range[0]) + std_range[0]
    )
    vars = stdevs ** 2
    covs = numpy.zeros((3, 3), float)
    numpy.fill_diagonal(covs, vars)
    cov_std = (cov_range[1] - cov_range[0]) / 4
    cov_mean = cov_std * 2 + cov_range[0]
    for row_idx, _ in enumerate(covs):
        for col_idx in range(row_idx + 1, len(covs)):
            cov = rand_gen.normal(loc=cov_mean, scale=cov_std)
            covs[row_idx][col_idx] = cov
            covs[col_idx][row_idx] = cov
    return covs


def gen_gaussian(
    dim: int,
    mean_range: Tuple[float, float] = (-1, 1),
    std_lim: Tuple[float, float] = (1, 1.75),
    cov_range: Tuple[float, float] = (-0.5, 0.5),
    prev: Optional[
        stats._multivariate.multivariate_normal_frozen  # pylint: disable=protected-access
    ] = None,
) -> stats._multivariate.multivariate_normal_frozen:
    """Generate a gaussian distribution"""
    means = gen_random_means(dim, mean_range, prev)
    covs = gen_cov_matrix(dim, std_lim, cov_range)
    return stats.multivariate_normal(mean=means, cov=covs)


def get_pdf_callable(
    dist: List[stats._multivariate.multivariate_normal_frozen], priors: List[float]
) -> Callable[[List[float]], float]:
    """Combine gaussians into a single distribution based on priors"""

    def pdf(samp: List[float]) -> float:
        raw_probs: List[float] = [d.pdf(samp) for d in dist]
        return sum([prior * raw for prior, raw in zip(priors, raw_probs)])

    return pdf


def gen_all_gaussians(
    dim: int, count: int
) -> List[
    stats._multivariate.multivariate_normal_frozen
]:  # pylint: disable=protected-access
    """Retrieve all distributions for question 2"""
    ret = list()
    for idx in range(count):
        if idx == 0:
            ret.append(gen_gaussian(dim))
        else:
            ret.append(gen_gaussian(dim, prev=ret[idx - 1]))
    return ret


def gen_dist_data(
    x_range: Tuple[float, float],
    x_cnt: int,
    dists: List[
        stats._multivariate.multivariate_normal_frozen
    ],  # pylint: disable=protected-access
) -> pandas.DataFrame:
    """Generate a dataframe containing an equal number of samples from each dist"""
    frame = pandas.DataFrame(columns=["dist", "sample", "prob"])
    step = (x_range[1] - x_range[0]) / x_cnt
    x_vals = numpy.arange(x_range[0], x_range[1], step)
    for idx, dist in enumerate(dists):
        sub = pandas.DataFrame(
            [[idx, x_val, dist.pdf(x_val)] for x_val in x_vals],
            columns=["dist", "sample", "prob"],
        )
        frame = frame.append(sub)
    return frame


def gen_ideal_dist_plot(frame: pandas.DataFrame) -> axes.Axes:
    """Create a plot to check that the distributions are ideally spaced"""
    return seaborn.lineplot(data=frame, x="sample", y="prob", hue="dist")
