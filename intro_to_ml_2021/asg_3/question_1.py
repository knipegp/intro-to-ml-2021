"""Functions for question 1"""
#!/usr/bin/env python3

from typing import Tuple, List, Dict, Union

import pandas
import numpy
from numpy import random
from numpy import linalg
from scipy import stats


def get_cube_verts(
    center: Tuple[float, float, float], length: float
) -> List[Tuple[float, float, float]]:
    """Return the vertices to the specified cube"""
    center_disp = length / 2
    verts: List[Tuple[float, float, float]] = list()
    for mult_x in [1, -1]:
        for mult_y in [1, -1]:
            for mult_z in [1, -1]:
                verts.append(
                    (
                        center_disp * mult_x + center[0],
                        center_disp * mult_y + center[1],
                        center_disp * mult_z + center[2],
                    )
                )

    return verts


def get_random_mean(
    center: Tuple[float, float, float] = (0, 0, 0), length: float = 2
) -> numpy.ndarray:
    """Generate a random mean vector"""
    verts = get_cube_verts(center, length)
    raw_choices = list(range(len(verts)))
    choice = random.choice(raw_choices, replace=False)
    return numpy.array(verts[choice])


def get_rand_cov(
    size: int, var_range: Tuple[float, float], eigval_range: Tuple[float, float]
) -> numpy.ndarray:
    """Generate a random covariance matrix with eigenvalues within the given interval"""
    while True:  # Forgive me
        # for _ in range(1000):
        raw_cov = (var_range[1] - var_range[0]) * random.random_sample(
            (size, size)
        ) + var_range[0]
        symm = numpy.tril(raw_cov, -1).T
        cov = numpy.tril(raw_cov, 0) + symm
        eigs = linalg.eigvals(cov)
        good_eigs = [False] * len(eigs)
        for idx, eigval in enumerate(eigs):
            if eigval < eigval_range[0] or eigval > eigval_range[1]:
                break
            else:
                good_eigs[idx] = True
        if False not in good_eigs:
            return cov
    # raise ValueError


# def choices(size: int, priors: List[float]) -> List[int]:
#     """Return a choice given priors"""
#     ranges: List[Tuple[float, float]] = list()
#     start = 0.0
#     choice_cnts: List[int] = [0] * len(priors)
#     for prior in priors:
#         ranges.append((start, prior + start))
#         start += prior
#     assert ranges[-1][1] <= 1.0
#     for _ in range(size):
#         rand_val: float = random.random_sample()
#         for choice, choice_range in enumerate(ranges):
#             if rand_val >= choice_range[0] and rand_val < choice_range[1]:
#                 break
#         assert rand_val >= choice_range[0] and rand_val < choice_range[1]
#         choice_cnts[choice] += 1
#     print(choice_cnts)
#     return choice_cnts


def gen_data(
    means: numpy.ndarray,
    covs: numpy.ndarray,
    priors: List[float],
    name_size: Dict[str, int],
) -> pandas.DataFrame:
    """Generate data sets"""
    sample_total = sum(name_size.values())
    choice_per_op = random.choice(list(range(len(priors))), sample_total, p=priors)
    single_choices = numpy.unique(choice_per_op)
    samples: Union[numpy.ndarray, None] = None
    for single_choice in single_choices:
        choice_cnt = numpy.where(choice_per_op == single_choice)[0].shape[0]
        class_samps = stats.multivariate_normal(
            mean=means[single_choice], cov=covs[single_choice]
        ).rvs(choice_cnt)
        samps_with_class = numpy.hstack(
            (class_samps, numpy.array([single_choice] * choice_cnt, ndmin=2).T)
        )
        if samples is None:
            samples = samps_with_class
        else:
            samples = numpy.append(samples, samps_with_class, axis=0)
    raw_sample_sets: List[str] = list()
    for set_name, set_size in name_size.items():
        raw_sample_sets += [set_name] * set_size
    sample_sets = numpy.array(raw_sample_sets).reshape((len(raw_sample_sets), 1))
    print(sample_sets.size)
    # for set_name, set_size in name_size.items():
    #     class_cnts = choices()
    #     stats.multivariate_normal()
