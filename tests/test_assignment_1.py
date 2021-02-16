#!/usr/bin/env python3

from typing import List

import numpy
import pytest
from intro_to_ml_2021.asg_1 import question_1, runner
from scipy import stats


@pytest.fixture(scope="session")
def dists() -> List[stats._multivariate.multivariate_normal_frozen]:
    dists = [
        stats.multivariate_normal(mean=runner.Means[0], cov=runner.Covs[0]),
        stats.multivariate_normal(mean=runner.Means[1], cov=runner.Covs[1]),
    ]
    return dists


@pytest.fixture(scope="session")
def labelled(dists) -> List[question_1.LabelledSample]:
    return question_1.gen_labelled_samples(
        runner.Priors,
        dists,
        100,
    )


def test_class_counts():
    counts = question_1.gen_label_cnt(runner.Priors, 100)
    assert counts[0] > counts[1]


def test_gen_samples(labelled):
    assert len(labelled) == 100
    assert len(labelled[0].samp) == len(runner.Means[0])


def test_roc_points(dists, labelled):
    points = runner.roc_points(labelled, 10, dists)
    assert numpy.array(points).shape[0] == 10


def test_thresh_alarm(labelled, dists):
    ratios = runner.get_ratios(labelled, dists)
    alarm_0 = runner.calc_thresh_alarm(0, ratios, labelled)
    alarm_1 = runner.calc_thresh_alarm(numpy.Inf, ratios, labelled)

    print(f"{alarm_0.false_prob} {alarm_0.true_prob}")
    assert round(alarm_0.false_prob) == round(alarm_0.true_prob) == 1
    assert round(alarm_1.false_prob) == round(alarm_1.true_prob) == 0
