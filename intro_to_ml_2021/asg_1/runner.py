"""Run functions to create answers for assignment 1"""
# /usr/bin/env python3
from dataclasses import dataclass
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
import numpy
import pandas
import seaborn
from scipy import stats

from intro_to_ml_2021.asg_1 import question_1

SAMPLECNT = 10000
Means = numpy.array([[-1.0, 1.0, -1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
Covs = numpy.array(
    [
        [[2.0, -0.5, 0.3, 0], [-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]],
        [[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]],
    ]
)
Priors = numpy.array([0.7, 0.3])


@dataclass
class AlarmProbs:
    true_1: float
    false_1: float
    false_0: float


def get_ratios(
    labelled: List[Tuple[float, int]],
    dists: List[
        stats._multivariate.multivariate_normal_frozen
    ],  # pylint: disable=protected-access
) -> List[float]:
    """Get the likelihood ratios for all samples"""
    norm_pdf_0 = [dists[0].pdf(samp[0]) for samp in labelled]
    norm_pdf_1 = [dists[1].pdf(samp[0]) for samp in labelled]
    return [prob_1 / prob_0 for prob_0, prob_1 in zip(norm_pdf_0, norm_pdf_1)]


def calc_thresh_alarm(
    thresh: float,
    ratios: List[float],
    labelled: List[Tuple[float, int]],
) -> AlarmProbs:
    """Get the true alarm and false alarm probabilities"""
    dec = [1 if ratio > thresh else 0 for ratio in ratios]
    l_1 = tuple(filter(lambda x: x[1][1] == 1, zip(dec, labelled)))
    l_0 = tuple(filter(lambda x: x[1][1] == 0, zip(dec, labelled)))
    prob_true_pos = len(tuple(filter(lambda pair: pair[0] == 1, l_1))) / len(l_1)
    prob_false_pos = len(tuple(filter(lambda pair: pair[0] == 1, l_0))) / len(l_0)
    prob_false_neg = len(tuple(filter(lambda pair: pair[0] == 0, l_1))) / len(l_1)
    return AlarmProbs(
        true_1=prob_true_pos, false_1=prob_false_pos, false_0=prob_false_neg
    )


def roc_points(
    labelled: List[Tuple[float, int]],
    threshes: numpy.ndarray,
    dists: List[
        stats._multivariate.multivariate_normal_frozen
    ],  # pylint: disable=protected-access
) -> Dict[float, AlarmProbs]:
    """Get all points for the ROC curve"""
    ratios = get_ratios(labelled, dists)
    return {thresh: calc_thresh_alarm(thresh, ratios, labelled) for thresh in threshes}


def get_minimum_error_threshold(points: Dict[float, AlarmProbs]) -> float:
    """Get the threshold value and ROC point for minimum error"""
    acc_thresh = list(points.keys())[0]
    acc = (
        points[acc_thresh].false_1 * Priors[0] + points[acc_thresh].false_0 * Priors[1]
    )
    for thresh in list(points.keys())[1:]:
        check = points[thresh].false_1 * Priors[0] + points[thresh].false_0 * Priors[1]
        if acc > check:
            acc_thresh = thresh
            acc = check
    return acc_thresh


def gen_question_1_fig() -> seaborn.FacetGrid:
    """Create the ROC curve figure"""
    dists: List[
        stats._multivariate.multivariate_normal_frozen
    ] = [  # pylint: disable=protected-access
        stats.multivariate_normal(mean=Means[0], cov=Covs[0]),
        stats.multivariate_normal(mean=Means[1], cov=Covs[1]),
    ]
    labelled: List[Tuple[float, int]] = question_1.gen_labelled_samples(
        Priors, dists, SAMPLECNT
    )
    prior_ratio = Priors[0] / Priors[1]
    threshes, ideal_idx = question_1.get_threshes(1000, prior_ratio)
    roc_data = roc_points(labelled, threshes, dists)
    best_thresh = get_minimum_error_threshold(roc_data)
    points = pandas.DataFrame(
        [
            [thresh, alarms.false_1, alarms.true_1, alarms.false_0]
            for thresh, alarms in roc_data.items()
        ],
        columns=["thresh", "false_1", "true_1", "false_0"],
    )
    grid = seaborn.FacetGrid(data=points)
    grid.map_dataframe(seaborn.lineplot, x="false_1", y="true_1")
    grid.set_axis_labels("P(D=0|L=1)", "P(D=1|L=1)")
    best_x = roc_data[best_thresh].false_1
    best_y = roc_data[best_thresh].true_1
    theo_thresh = threshes[ideal_idx]
    theo_x = roc_data[theo_thresh].false_1
    theo_y = roc_data[theo_thresh].true_1
    grid.axes[0][0].text(best_x, best_y, f"best calculated threshold = {best_thresh}")
    grid.axes[0][0].text(theo_x, theo_y, f"theoretical threshold = {prior_ratio}")
    return grid


fig = gen_question_1_fig()

plt.show()
