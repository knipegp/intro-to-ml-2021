"""Run functions to create answers for assignment 1"""
# /usr/bin/env python3
from dataclasses import dataclass
import functools
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy
import pandas
import seaborn
from scipy import stats
from matplotlib import pyplot

from intro_to_ml_2021.asg_1 import question_1, question_2, question_3

SAMPLECNT = 10000
MEANS = numpy.array([[-1.0, 1.0, -1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
COVSA = numpy.array(
    [
        [[2.0, -0.5, 0.3, 0], [-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]],
        [[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]],
    ]
)
COVSB = numpy.array([numpy.diagflat(numpy.diagonal(arr)) for arr in COVSA])
PRIORS_1 = [0.7, 0.3]
PRIORS_2 = [0.3, 0.3, 0.4]

LOSS_MATS = [[[0, 1, 10], [1, 0, 10], [1, 1, 0]], [[0, 1, 100], [1, 0, 100], [1, 1, 0]]]

PHONE_ROOT = Path("~/Downloads/ml_dataset/UCI HAR Dataset").expanduser()


@dataclass
class AlarmProbs:
    """Collection of alarm types for a threshold"""

    true_1: float
    false_1: float
    false_0: float


def get_ratios(
    labelled: List[Tuple[List[float], int]],
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
    labelled: List[Tuple[List[float], int]],
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
    labelled: List[Tuple[List[float], int]],
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
        points[acc_thresh].false_1 * PRIORS_1[0]
        + points[acc_thresh].false_0 * PRIORS_1[1]
    )
    for thresh in list(points.keys())[1:]:
        check = (
            points[thresh].false_1 * PRIORS_1[0] + points[thresh].false_0 * PRIORS_1[1]
        )
        if acc > check:
            acc_thresh = thresh
            acc = check
    return acc_thresh


def gen_question_1_fig(
    priors: List[float],
    labelled: List[Tuple[List[float], int]],
    dists: List[stats._multivariate.multivariate_normal_frozen],
) -> seaborn.FacetGrid:
    """Create the ROC curve figure"""
    prior_ratio = priors[0] / priors[1]
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


def run_question_1() -> List[seaborn.FacetGrid]:
    """Create graphs for both parts A and B"""
    dists_base: List[
        stats._multivariate.multivariate_normal_frozen
    ] = [  # pylint: disable=protected-access
        stats.multivariate_normal(mean=MEANS[0], cov=COVSA[0]),
        stats.multivariate_normal(mean=MEANS[1], cov=COVSA[1]),
    ]
    labelled: List[Tuple[List[float], int]] = question_1.gen_labelled_samples(
        PRIORS_1, dists_base, SAMPLECNT
    )
    bad_dists: List[
        stats._multivariate.multivariate_normal_frozen  # pylint: disable=protected-access
    ] = [
        stats.multivariate_normal(mean=MEANS[0], cov=COVSB[0]),
        stats.multivariate_normal(mean=MEANS[1], cov=COVSB[1]),
    ]
    return [
        gen_question_1_fig(PRIORS_1, labelled, dists_base),
        gen_question_1_fig(PRIORS_1, labelled, bad_dists),
    ]


def gen_question_2_dists(
    gaus: List[stats._multivariate.multivariate_normal_frozen],
) -> List[Tuple[List[float], int]]:
    """Generated labelled data for question 2"""
    labelled = question_1.gen_labelled_samples(
        [PRIORS_2[0], PRIORS_2[1], PRIORS_2[2] / 2, PRIORS_2[2] / 2], gaus, SAMPLECNT
    )
    for idx, elem in enumerate(labelled):
        if elem[1] == 3:
            new_elem = (elem[0], 2)
            labelled[idx] = new_elem
    return labelled


def get_decision(
    pdfs: List[Callable[[List[float]], float]], sample: List[float]
) -> int:
    """Stop"""
    like_prior = numpy.array([pdf(sample) for pdf in pdfs])
    evidence = sum(like_prior)
    errors = [1 - lp / evidence for lp in like_prior]
    zipped = list(enumerate(errors))
    min_error: Tuple[int, Any] = functools.reduce(
        lambda accum, elem: elem if elem[1] < accum[1] else accum, zipped[1:], zipped[0]
    )
    return min_error[0]


def get_lossy_decision(
    pdfs: List[Callable[[List[float]], float]],
    loss_mat: List[List[int]],
    sample: List[float],
) -> int:
    like_prior = [pdf(sample) for pdf in pdfs]
    evidence = sum(like_prior)
    errors: List[float] = list()
    for loss in loss_mat:
        errors.append(
            sum([elem[1] * elem[0] / evidence for elem in zip(like_prior, loss)])
        )
    zipped = list(enumerate(errors))
    min_error: Tuple[int, Any] = functools.reduce(
        lambda accum, elem: elem if elem[1] < accum[1] else accum, zipped[1:], zipped[0]
    )
    return min_error[0]


def get_question_2_data() -> Tuple[
    List[Tuple[List[float], int]], List[Callable[[List[float]], float]]
]:
    gaus = question_2.gen_all_gaussians(3, 4)
    labelled = gen_question_2_dists(gaus)
    combined_dists = [
        question_2.get_pdf_callable(gaus[0:1], PRIORS_2[0:1]),
        question_2.get_pdf_callable(gaus[1:2], PRIORS_2[1:2]),
        question_2.get_pdf_callable(gaus[2:], PRIORS_2[2:]),
    ]
    return (labelled, combined_dists)


def gen_question_2_plot(labelled: List[Tuple[List[float], int]], decisions: List[int]):
    frame = pandas.DataFrame(
        [
            [samp[0][0], samp[0][1], samp[0][2], samp[1], dec]
            for samp, dec in zip(labelled, decisions)
        ],
        columns=["x", "y", "z", "label", "dec"],
    )

    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection="3d")
    for class_label, shape in enumerate(["o", "^", "s"]):
        right = frame.query(f"label == {class_label} and label == dec")
        wrong = frame.query(f"label == {class_label} and label != dec")
        print(
            f"class {class_label}, error rate {wrong.size / (right.size + wrong.size)}"
        )
        ax.scatter(right.x, right.y, right.z, marker=shape, c="g")
        ax.scatter(wrong.x, wrong.y, wrong.z, marker=shape, c="r")
    return fig


def run_question_2():
    labelled, combined_dists = get_question_2_data()
    decs_a = [get_decision(combined_dists, samp[0]) for samp in labelled]
    decs_b = [
        get_lossy_decision(combined_dists, LOSS_MATS[0], samp[0]) for samp in labelled
    ]
    decs_c = [
        get_lossy_decision(combined_dists, LOSS_MATS[1], samp[0]) for samp in labelled
    ]
    # gen_question_2_plot(labelled, decs_a)
    # gen_question_2_plot(labelled, decs_b)
    gen_question_2_plot(labelled, decs_c)


def get_question_3_b_data() -> Tuple[numpy.ndarray, numpy.ndarray, pandas.DataFrame]:
    """Put all question 3 phone data in a Pandas dataframe"""
    int_to_label = question_3.get_info(Path(PHONE_ROOT, "activity_labels.txt"))
    feat_names = question_3.get_info(Path(PHONE_ROOT, "features.txt"))
    frame = pandas.DataFrame(columns=list(feat_names.values()))
    subjects = numpy.array(list())
    labels = numpy.array(list(), dtype=int)
    for suf in ["train", "test"]:
        subjects = numpy.append(
            subjects,
            numpy.loadtxt(Path(PHONE_ROOT, suf, f"subject_{suf}.txt"), dtype=int),
        )
        labels = numpy.append(
            labels, numpy.loadtxt(Path(PHONE_ROOT, suf, f"y_{suf}.txt"), dtype=int)
        )
        features = numpy.loadtxt(Path(PHONE_ROOT, suf, f"X_{suf}.txt"), dtype=float)
        new_frame = pandas.DataFrame(
            features,
            columns=list(feat_names.values()),
        )
        frame = frame.append(new_frame, ignore_index=True)
    return subjects, labels, frame


def get_mean_vectors(
    labels: numpy.ndarray, frame: pandas.DataFrame
) -> List[stats._multivariate.multivariate_normal_frozen]:
    """Get a mean vector for each label"""
    dists = list()
    for lab in numpy.unique(labels):
        idxs = numpy.where(labels == lab)
        for idx in idxs:
            mean = frame.loc[idx].mean().values
            cov = frame.loc[idx].cov().values
            dists.append(
                stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True)
            )

    return dists


_, labels, frame = get_question_3_b_data()
dists = get_mean_vectors(labels, frame)
print(dists)
# dist = stats.multivariate_normal()
# priors = numpy.bincount(labels)[1:] / labels.shape[0]
# print(priors)
# print(frame.mean())
# print(frame.cov())
