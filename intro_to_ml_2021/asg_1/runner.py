"""Run functions to create answers for assignment 1"""
# /usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy
import pandas
import seaborn
from intro_to_ml_2021.asg_1 import question_1, question_2, question_3
from matplotlib import pyplot
from scipy import stats

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

ALPHA = 0.195


@dataclass
class AlarmProbs:
    """Collection of alarm types for a threshold"""

    true_1: float
    false_1: float
    false_0: float


def get_ratios(
    samples: numpy.ndarray,
    dists: List[
        stats._multivariate.multivariate_normal_frozen  # pylint: disable=protected-access
    ],
) -> numpy.ndarray:
    """Get the likelihood ratios for all samples"""
    norm_pdf_0 = numpy.array([dists[0].pdf(samp) for samp in samples])
    norm_pdf_1 = numpy.array([dists[1].pdf(samp) for samp in samples])
    return norm_pdf_1 / norm_pdf_0


def calc_thresh_alarm(
    thresh: float,
    ratios: numpy.ndarray,
    lab_1_idxs: numpy.ndarray,
    lab_0_idxs: numpy.ndarray,
) -> AlarmProbs:
    """Get the true alarm and false alarm probabilities"""
    dec = numpy.where(ratios > thresh, 1, 0)

    dec_any_lab_1 = dec[lab_1_idxs]
    dec_1_lab_1_idxs = numpy.where(dec_any_lab_1 == 1)[0]
    dec_0_lab_1_idxs = numpy.where(dec_any_lab_1 == 0)[0]

    dec_any_lab_0 = dec[lab_0_idxs]
    dec_1_lab_0_idxs = numpy.where(dec_any_lab_0 == 1)[0]

    prob_true_pos = dec_1_lab_1_idxs.shape[0] / lab_1_idxs.shape[0]
    prob_false_pos = dec_1_lab_0_idxs.shape[0] / lab_0_idxs.shape[0]
    prob_false_neg = dec_0_lab_1_idxs.shape[0] / lab_1_idxs.shape[0]
    return AlarmProbs(
        true_1=prob_true_pos, false_1=prob_false_pos, false_0=prob_false_neg
    )


def roc_points(
    samples: numpy.ndarray,
    labels: numpy.ndarray,
    threshes: numpy.ndarray,
    dists: List[
        stats._multivariate.multivariate_normal_frozen  # pylint: disable=protected-access
    ],
) -> Dict[float, AlarmProbs]:
    """Get all points for the ROC curve"""
    ratios = get_ratios(samples, dists)
    lab_1_idxs = numpy.where(labels == 1)[0]
    lab_0_idxs = numpy.where(labels == 0)[0]
    return {
        thresh: calc_thresh_alarm(thresh, ratios, lab_1_idxs, lab_0_idxs)
        for thresh in threshes
    }


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


def gen_question_1_fig_data(
    prior_ratio: float,
    samples: numpy.ndarray,
    labels: numpy.ndarray,
    dists: List[
        stats._multivariate.multivariate_normal_frozen  # pylint: disable=protected-access
    ],
) -> Tuple[int, Dict[float, AlarmProbs], numpy.ndarray, pandas.DataFrame]:
    threshes, ideal_idx = question_1.get_threshes(1000, prior_ratio)
    roc_data = roc_points(samples, labels, threshes, dists)
    points = pandas.DataFrame(
        [
            [thresh, alarms.false_1, alarms.true_1, alarms.false_0]
            for thresh, alarms in roc_data.items()
        ],
        columns=["thresh", "false_1", "true_1", "false_0"],
    )
    return ideal_idx, roc_data, threshes, points


def gen_question_1_fig(
    prior_ratio: float,
    ideal_thresh_idx: int,
    roc_data: Dict[float, AlarmProbs],
    threshes: numpy.ndarray,
    points: pandas.DataFrame,
) -> seaborn.FacetGrid:
    """Create the ROC curve figure"""
    best_thresh = get_minimum_error_threshold(roc_data)
    grid = seaborn.FacetGrid(data=points)
    grid.map_dataframe(seaborn.lineplot, x="false_1", y="true_1")
    grid.set_axis_labels("P(D=0|L=1)", "P(D=1|L=1)")
    best_x = roc_data[best_thresh].false_1
    best_y = roc_data[best_thresh].true_1
    theo_thresh = threshes[ideal_thresh_idx]
    theo_x = roc_data[theo_thresh].false_1
    theo_y = roc_data[theo_thresh].true_1
    grid.axes[0][0].text(best_x, best_y, f"best calculated threshold = {best_thresh}")
    grid.axes[0][0].text(theo_x, theo_y, f"theoretical threshold = {prior_ratio}")
    return grid


def run_question_1() -> List[seaborn.FacetGrid]:
    """Create graphs for both parts A and B"""
    dists_base: List[
        stats._multivariate.multivariate_normal_frozen  # pylint: disable=protected-access
    ] = [
        stats.multivariate_normal(mean=MEANS[0], cov=COVSA[0]),
        stats.multivariate_normal(mean=MEANS[1], cov=COVSA[1]),
    ]
    samples, labels = question_1.gen_labelled_samples(PRIORS_1, dists_base, SAMPLECNT)
    bad_dists: List[
        stats._multivariate.multivariate_normal_frozen  # pylint: disable=protected-access
    ] = [
        stats.multivariate_normal(mean=MEANS[0], cov=COVSB[0]),
        stats.multivariate_normal(mean=MEANS[1], cov=COVSB[1]),
    ]
    prior_ratio = PRIORS_1[0] / PRIORS_1[1]
    ideal_idx_1, roc_data_1, threshes_1, points_1 = gen_question_1_fig_data(
        prior_ratio, samples, labels, dists_base
    )
    ideal_idx_2, roc_data_2, threshes_2, points_2 = gen_question_1_fig_data(
        prior_ratio, samples, labels, bad_dists
    )

    return [
        gen_question_1_fig(prior_ratio, ideal_idx_1, roc_data_1, threshes_1, points_1),
        gen_question_1_fig(prior_ratio, ideal_idx_2, roc_data_2, threshes_2, points_2),
    ]


def gen_question_2_dists(
    gaus: List[
        stats._multivariate.multivariate_normal_frozen  # pylint: disable=protected-access
    ],
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Generated labelled data for question 2"""
    samples, labels = question_1.gen_labelled_samples(
        [PRIORS_2[0], PRIORS_2[1], PRIORS_2[2] / 2, PRIORS_2[2] / 2], gaus, SAMPLECNT
    )
    for idx in numpy.where(labels == 3):
        labels[idx] = 2
    return samples, labels


def get_decision(
    pdfs: Dict[int, Callable[[List[float]], float]], sample: List[float]
) -> int:
    """Stop"""
    label_pdf = numpy.array([[label, pdf(sample)] for label, pdf in pdfs.items()])
    labels = label_pdf[:, 0]
    like_prior = label_pdf[:, 1]
    evidence = numpy.sum(like_prior)
    errors = numpy.array([1 - lp / evidence for lp in like_prior])
    dec_idx = errors.argmin()
    return labels[dec_idx]


def get_lossy_decision(
    pdfs: Dict[int, Callable[[List[float]], float]],
    loss_mat: List[List[int]],
    sample: List[float],
) -> int:
    label_pdf = numpy.array([[label, pdf(sample)] for label, pdf in pdfs.items()])
    labels = label_pdf[:, 0]
    like_prior = label_pdf[:, 1]
    evidence = numpy.sum(like_prior)
    errors = numpy.array(
        [sum(loss_vec * like_prior / evidence) for loss_vec in loss_mat]
    )
    dec_idx = errors.argmin()
    return labels[dec_idx]


def get_question_2_data() -> Tuple[
    numpy.ndarray, numpy.ndarray, Dict[int, Callable[[List[float]], float]]
]:
    gaus = question_2.gen_all_gaussians(3, 4)
    samples, labels = gen_question_2_dists(gaus)
    combined_dists = {
        0: question_2.get_pdf_callable(gaus[0:1], PRIORS_2[0:1]),
        1: question_2.get_pdf_callable(gaus[1:2], PRIORS_2[1:2]),
        2: question_2.get_pdf_callable(gaus[2:], PRIORS_2[2:]),
    }
    return (samples, labels, combined_dists)


def gen_question_2_plot(
    samples: numpy.ndarray, labels: numpy.ndarray, decisions: numpy.ndarray
):
    """Generate a plot for question 2"""
    frame = pandas.DataFrame(
        numpy.hstack(
            (
                samples,
                labels.reshape((labels.shape[0], 1)),
                decisions.reshape((decisions.shape[0], 1)),
            )
        ),
        columns=["x", "y", "z", "label", "dec"],
    )

    fig = pyplot.figure()
    axes = fig.add_subplot(111, projection="3d")
    for class_label, shape in enumerate(["o", "^", "s"]):
        right = frame.query(f"label == {class_label} and label == dec")
        wrong = frame.query(f"label == {class_label} and label != dec")
        print(
            f"class {class_label}, error rate {wrong.size / (right.size + wrong.size)}"
        )
        axes.scatter(right.x, right.y, right.z, marker=shape, c="g")
        axes.scatter(wrong.x, wrong.y, wrong.z, marker=shape, c="r")
    return fig


def run_question_2():
    """Run question 2"""
    samples, labels, combined_dists = get_question_2_data()
    decs_a = numpy.array([get_decision(combined_dists, samp) for samp in samples])
    decs_b = numpy.array(
        [get_lossy_decision(combined_dists, LOSS_MATS[0], samp) for samp in samples]
    )
    decs_c = numpy.array(
        [get_lossy_decision(combined_dists, LOSS_MATS[1], samp) for samp in samples]
    )
    gen_question_2_plot(samples, labels, decs_a)
    gen_question_2_plot(samples, labels, decs_b)
    gen_question_2_plot(samples, labels, decs_c)


def get_question_3_b_data() -> Tuple[numpy.ndarray, numpy.ndarray, pandas.DataFrame]:
    """Put all question 3 phone data in a Pandas dataframe"""
    feat_names = question_3.get_info(Path(PHONE_ROOT, "features.txt"))
    frame = pandas.DataFrame(columns=feat_names.values())
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


def get_q_3_accuracy(
    labels: numpy.ndarray, frame: pandas.DataFrame, alpha: float
) -> float:
    """Get the accuracy of classification for the given data"""
    dists = question_3.get_mean_vectors(labels, frame, alpha)
    bins = numpy.bincount(labels)
    priors = bins[numpy.nonzero(bins)] / sum(bins)
    callables = {
        lab: question_2.get_pdf_callable([d], [p])
        for lab, d, p in zip(numpy.unique(labels), dists, priors)
    }
    right = 0
    for idx, label in enumerate(labels):
        dec = get_decision(callables, frame.loc[idx].values)
        if dec == label:
            right += 1
    return right / sum(bins)


def run_question_3_b() -> float:
    """Return the correct classification rate"""
    _, labels, frame = get_question_3_b_data()
    return get_q_3_accuracy(labels, frame, ALPHA)


def run_question_3_a() -> float:
    """Run question 3 a"""
    frame: pandas.DataFrame = pandas.read_csv(
        Path("~/Downloads/winequality-white.csv").expanduser(), sep=";"
    )
    quality: numpy.ndarray = frame.pop("quality").values
    return get_q_3_accuracy(quality, frame, 0.3)


run_question_1()
pyplot.show()
