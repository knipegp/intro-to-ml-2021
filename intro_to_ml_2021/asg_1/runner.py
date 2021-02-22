"""Run functions to create answers for assignment 1"""
# /usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union, Optional

import numpy
import pandas
import seaborn
from intro_to_ml_2021.asg_1 import question_1, question_2, question_3
from matplotlib import pyplot, axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
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

SHAPES = ["o", "^", "s", "x", "*", "1", "+"]

MEANS2 = [
    [-0.30450554, 0.93604865, -0.22862054],
    [3.55235413, 3.68764659, 2.84013595],
    [5.47162041, 6.50151399, 5.52557088],
    [7.73101012, 8.9361048, 9.19035942],
]
COVS2 = [
    [
        [2.38901889, -0.35284848, -0.11407521],
        [-0.35284848, 1.28997632, -0.01698769],
        [-0.11407521, -0.01698769, 2.72670782],
    ],
    [
        [1.61264, -0.25857834, -0.01816494],
        [-0.25857834, 1.0051108, -0.83846096],
        [-0.01816494, -0.83846096, 1.93678259],
    ],
    [
        [2.09689469, -0.09006346, -0.04448728],
        [-0.09006346, 2.78146741, -0.16375982],
        [-0.04448728, -0.16375982, 1.98309164],
    ],
    [
        [2.7585717, 0.37661328, 0.26309899],
        [0.37661328, 2.62633189, -0.16691706],
        [0.26309899, -0.16691706, 1.14029808],
    ],
]

LOSS_3A = [
    [0, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 1],
]


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
    return numpy.array(norm_pdf_1 / norm_pdf_0)


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


def print_question_1_error(
    roc_data: Dict[float, AlarmProbs],
    best_thresh: float,
    ideal_thresh: float,
    priors: List[float],
):
    """Print the error rate for a question 1 distribution"""
    ideal_error = (
        roc_data[ideal_thresh].false_1 * priors[0]
        + roc_data[ideal_thresh].false_0 * priors[1]
    )
    best_error = (
        roc_data[best_thresh].false_1 * priors[0]
        + roc_data[best_thresh].false_0 * priors[1]
    )
    print(
        f"theoretical threshold true positive rate"
        f"{roc_data[ideal_thresh].true_1} false positive rate"
        f"{roc_data[ideal_thresh].false_1} error rate {ideal_error}"
    )
    print(
        f"calculated threshold true positive rate {roc_data[best_thresh].true_1}"
        f"false positive rate {roc_data[best_thresh].false_1} error rate"
        f"{best_error}"
    )


def gen_question_1_fig_data(  # pylint: disable=too-many-arguments
    priors: List[float],
    samples: numpy.ndarray,
    labels: numpy.ndarray,
    dists: List[
        stats._multivariate.multivariate_normal_frozen  # pylint: disable=protected-access
    ],
    threshes: numpy.ndarray,
    ideal_idx: int,
) -> axes.Axes:
    """Generate the data for a figure"""
    prior_ratio = priors[0] / priors[1]
    roc_data = roc_points(samples, labels, threshes, dists)
    points = pandas.DataFrame(
        [
            [thresh, alarms.false_1, alarms.true_1, alarms.false_0]
            for thresh, alarms in roc_data.items()
        ],
        columns=["thresh", "false_1", "true_1", "false_0"],
    )
    best_thresh = get_minimum_error_threshold(roc_data)
    print_question_1_error(roc_data, best_thresh, threshes[ideal_idx], priors)
    return gen_question_1_fig(
        prior_ratio, best_thresh, ideal_idx, roc_data, threshes, points
    )


def gen_question_1_fig(  # pylint: disable=too-many-arguments
    prior_ratio: float,
    best_thresh: float,
    ideal_thresh_idx: int,
    roc_data: Dict[float, AlarmProbs],
    threshes: numpy.ndarray,
    points: pandas.DataFrame,
) -> axes.Axes:
    """Create the ROC curve figure"""
    pyplot.figure()
    plot: axes.Axes = seaborn.lineplot(data=points, x="false_1", y="true_1")
    seaborn.scatterplot(
        data=points.where(points.thresh == best_thresh),
        x="false_1",
        y="true_1",
        c=["green"],
        ax=plot,
    )
    seaborn.scatterplot(
        data=points.where(points.thresh == prior_ratio),
        x="false_1",
        y="true_1",
        c=["red"],
        ax=plot,
    )
    plot.set_xlabel("P(D=0|L=1)")
    plot.set_ylabel("P(D=1|L=1)")
    best_args = [
        roc_data[best_thresh].false_1,
        roc_data[best_thresh].true_1,
        f"best calculated threshold = {round(best_thresh, 3)}",
    ]
    theo_thresh = threshes[ideal_thresh_idx]
    theo_args = [
        roc_data[theo_thresh].false_1,
        roc_data[theo_thresh].true_1,
        f"theoretical threshold = {round(prior_ratio, 3)}",
    ]
    bias = 0.01
    plot.axvline(best_args[0], color="green")
    plot.axhline(best_args[1], color="green")
    plot.axvline(theo_args[0], color="red")
    plot.axhline(theo_args[1], color="red")
    if abs(best_args[1] - theo_args[1]) < bias:
        top = max(best_args[1], theo_args[1])
        if top == best_args[1]:
            best_args[1] += bias
            theo_args[1] -= bias
        else:
            best_args[1] -= bias
            theo_args[1] += bias
    plot.text(best_args[0], best_args[1], best_args[2])
    plot.text(theo_args[0], theo_args[1], theo_args[2])
    plot.set_title("ROC Curve")
    return plot


def run_question_1() -> List[axes.Axes]:
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
    threshes, ideal_idx = question_1.get_threshes(1000, prior_ratio)
    fig_a = gen_question_1_fig_data(
        PRIORS_1, samples, labels, dists_base, threshes, ideal_idx
    )
    fig_b = gen_question_1_fig_data(
        PRIORS_1, samples, labels, bad_dists, threshes, ideal_idx
    )

    return [fig_a, fig_b]


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
) -> Tuple[int, float]:
    """Stop"""
    label_pdf = numpy.array([[label, pdf(sample)] for label, pdf in pdfs.items()])
    labels = label_pdf[:, 0]
    like_prior = label_pdf[:, 1]
    evidence = numpy.sum(like_prior)
    errors = numpy.array([1 - lp / evidence for lp in like_prior])
    dec_idx = errors.argmin()
    return labels[dec_idx], errors[dec_idx] * evidence


def get_lossy_decision(
    pdfs: Dict[int, Callable[[List[float]], float]],
    loss_mat: List[List[int]],
    sample: List[float],
) -> Tuple[int, float]:
    """Return the minimum error decision with the given loss matrix"""
    label_pdf = numpy.array([[label, pdf(sample)] for label, pdf in pdfs.items()])
    labels = label_pdf[:, 0]
    like_prior = label_pdf[:, 1]
    evidence = numpy.sum(like_prior)
    errors = numpy.array(
        [sum(numpy.multiply(loss_vec, like_prior) / evidence) for loss_vec in loss_mat]
    )
    print(f"{errors}")
    dec_idx = errors.argmin()
    return labels[dec_idx], errors[dec_idx] * evidence


def print_question_2_dists(
    dists: List[
        stats._multivariate.multivariate_normal_frozen  # pylint: disable=protected-access
    ],
):
    """Print the means vectos and covariance matrices"""
    for idx, dist in enumerate(dists):
        print(f"idx {idx} mean {dist.mean} cov {dist.cov}")


def get_question_2_data(
    means: Optional[List[List[float]]] = None,  # pylint: disable=unsubscriptable-object
    covs: Optional[  # pylint: disable=unsubscriptable-object
        List[List[List[float]]]
    ] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray, Dict[int, Callable[[List[float]], float]]]:
    """Generate data for question 2"""
    if means is None or covs is None:
        gaus = question_2.gen_all_gaussians(3, 4)
    else:
        gaus = list()
        for mean, cov in zip(means, covs):
            gaus.append(stats.multivariate_normal(mean=mean, cov=cov))
    print_question_2_dists(gaus)
    samples, labels = gen_question_2_dists(gaus)
    combined_dists = {
        0: question_2.get_pdf_callable(gaus[0:1], PRIORS_2[0:1]),
        1: question_2.get_pdf_callable(gaus[1:2], PRIORS_2[1:2]),
        2: question_2.get_pdf_callable(gaus[2:], PRIORS_2[2:]),
    }
    return (samples, labels, combined_dists)


def gen_question_2_plot(
    samples: numpy.ndarray,
    labels: numpy.ndarray,
    decisions: numpy.ndarray,
    axis_labels: Optional[List[str]] = None,  # pylint: disable=unsubscriptable-object
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
    plot: Axes3D = fig.add_subplot(111, projection="3d")
    for idx, class_label in enumerate(numpy.unique(labels)):
        shape = SHAPES[idx]
        right = frame.query(f"label == {class_label} and label == dec")
        wrong = frame.query(f"label == {class_label} and label != dec")
        plot.scatter(right.x, right.y, right.z, marker=shape, c="g")
        plot.scatter(wrong.x, wrong.y, wrong.z, marker=shape, c="r")
    if axis_labels is None:
        plot.set_xlabel(axis_labels[0])
        plot.set_ylabel(axis_labels[1])
        plot.set_zlabel(axis_labels[2])
    return fig


def print_conf_matrix(labels: numpy.ndarray, decs: numpy.ndarray):
    """Print the confidence matrix"""
    label_cnt = numpy.unique(labels).shape[0]
    conf_matrix = numpy.zeros((label_cnt, label_cnt), dtype=int)
    for lab_idx, lab in enumerate(numpy.unique(labels)):
        lab_idxs = numpy.where(labels == lab)[0]
        for dec_idx, dec in enumerate(numpy.unique(labels)):
            conf_matrix[dec_idx][lab_idx] += numpy.where(decs[lab_idxs] == dec)[0].shape

    print(conf_matrix)


def run_question_2():
    """Run question 2"""
    samples, labels, combined_dists = get_question_2_data(means=MEANS2, covs=COVS2)
    dec_risk_a = [get_decision(combined_dists, samp) for samp in samples]
    risks_a = numpy.array([tup[1] for tup in dec_risk_a])
    print(risks_a.sum())
    decs_a = numpy.array([tup[0] for tup in dec_risk_a])
    print_conf_matrix(labels, decs_a)
    dec_risk_b = numpy.array(
        [get_lossy_decision(combined_dists, LOSS_MATS[0], samp) for samp in samples]
    )
    risks_b = numpy.array([tup[1] for tup in dec_risk_b])
    print(risks_b.sum())
    decs_b = numpy.array([tup[0] for tup in dec_risk_b])
    print_conf_matrix(labels, decs_b)
    dec_risk_c = numpy.array(
        [get_lossy_decision(combined_dists, LOSS_MATS[1], samp) for samp in samples]
    )
    risks_c = numpy.array([tup[1] for tup in dec_risk_c])
    print(risks_c.sum())
    decs_c = numpy.array([tup[0] for tup in dec_risk_c])
    print_conf_matrix(labels, decs_c)
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
) -> Tuple[float, numpy.ndarray]:
    """Get the accuracy of classification for the given data"""
    dists = question_3.get_mean_vectors(labels, frame, alpha)
    bins = numpy.bincount(labels)
    priors = bins[numpy.nonzero(bins)] / sum(bins)
    callables = {
        lab: question_2.get_pdf_callable([d], [p])
        for lab, d, p in zip(numpy.unique(labels), dists, priors)
    }
    decisions = numpy.array(
        [
            get_decision(callables, frame.loc[idx].values)[0]
            for idx, _ in enumerate(labels)
        ]
    )
    print_conf_matrix(labels, decisions)
    right = 0
    for label in numpy.unique(labels):
        lab_idxs = numpy.where(labels == label)[0]
        right += numpy.where(decisions[lab_idxs] == label)[0].shape[0]

    return right / sum(bins), decisions


def test_normal(frame: pandas.DataFrame):
    "Return the p-value for the isnormal test"
    features: List[
        List[Union[str, float]]  # pylint: disable=unsubscriptable-object
    ] = list()
    for col in frame.columns.to_numpy():
        if frame[col].ndim == 1:
            features.append([col, stats.normaltest(frame[col].values)[1]])
    features = sorted(features, key=lambda elem: elem[1], reverse=True)
    frame = pandas.DataFrame(
        features[:6],
        columns=["feature", "stat"],
    )
    plot: axes.Axes = seaborn.barplot(data=frame, x="feature", y="stat")
    plot.set_ylabel("p-value")
    print(frame)


def run_question_3_b() -> float:
    """Return the correct classification rate"""
    _, labels, frame = get_question_3_b_data()
    test_normal(frame)
    acc, dec = get_q_3_accuracy(labels, frame, ALPHA)
    gen_question_2_plot(
        numpy.array(
            frame[
                ["tBodyAcc-mean()-X", "tBodyAcc-mean()-Y", "tBodyAcc-mean()-Z"]
            ].values
        ),
        labels,
        dec,
        ["tBodyAcc-mean()-X", "tBodyAcc-mean()-Y", "tBodyAcc-mean()-Z"],
    )
    return acc


def run_question_3_a() -> float:
    """Run question 3 a"""
    frame: pandas.DataFrame = pandas.DataFrame(
        pandas.read_csv(Path("~/Downloads/winequality-white.csv").expanduser(), sep=";")
    )
    quality: numpy.ndarray = numpy.array(frame.pop("quality").values)
    print(numpy.unique(quality))
    # test_normal(frame)
    acc, dec = get_q_3_accuracy(quality, frame, 0.3)
    gen_question_2_plot(
        numpy.array(frame[["citric acid", "pH", "alcohol"]].values),
        quality,
        dec,
        ["citric acid", "pH", "alcohol"],
    )
    return acc
