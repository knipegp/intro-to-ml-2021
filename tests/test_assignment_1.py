# pylint: disable=missing-docstring
# /usr/bin/env python3

from typing import List, Tuple

import numpy
import pytest
from intro_to_ml_2021.asg_1 import question_1, question_2, runner
from scipy import stats


@pytest.fixture(scope="session")
def dists() -> List[stats._multivariate.multivariate_normal_frozen]:
    dist_list = [
        stats.multivariate_normal(mean=runner.MEANS[0], cov=runner.COVSA[0]),
        stats.multivariate_normal(mean=runner.MEANS[1], cov=runner.COVSA[1]),
    ]
    return dist_list


@pytest.fixture(scope="session")
def labelled(dists) -> List[Tuple[List[float], int]]:
    return question_1.gen_labelled_samples(
        runner.PRIORS_1,
        dists,
        100,
    )


def test_class_counts():
    counts = question_1.gen_label_cnt(runner.PRIORS_1, 100)
    assert counts[0] > counts[1]


def test_gen_samples(labelled: List[Tuple[List[float], int]]):
    assert len(labelled) == 100
    assert len(labelled[0][0]) == len(runner.MEANS[0])


def test_roc_points(dists, labelled):
    threshes = question_1.get_threshes(10, 2)[0]
    points = runner.roc_points(labelled, threshes, dists)
    assert len(points) == 10


def test_thresh_alarm(labelled, dists):
    ratios = runner.get_ratios(labelled, dists)
    alarm_0 = runner.calc_thresh_alarm(0, ratios, labelled)
    alarm_1 = runner.calc_thresh_alarm(numpy.Inf, ratios, labelled)

    assert round(alarm_0.false_1) == round(alarm_0.true_1) == 1
    assert round(alarm_1.false_1) == round(alarm_1.true_1) == 0


def test_gen_gaussian():
    gau = question_2.gen_gaussian(3)
    assert len(gau.mean) == 3


def test_gen_dist_data():
    gaus = question_2.gen_all_gaussians(3, 3)
    frame = question_2.gen_dist_data((0, 10), 100, gaus)
    assert len(frame) == 300
