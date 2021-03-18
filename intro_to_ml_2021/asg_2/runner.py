"""Run assignment 2"""
#!/usr/bin/env python3
from typing import Dict, List, Optional

import numpy
import seaborn
import pandas
from sklearn.preprocessing import PolynomialFeatures
from intro_to_ml_2021.asg_2 import question_1, question_2
from matplotlib import axes, pyplot
from scipy import stats
from tqdm import tqdm

PRIORS: List[float] = [0.325, 0.325, 0.35]
MEANS: numpy.ndarray = numpy.array([[3, 0], [0, 3], [2, 2]])
COVS: numpy.ndarray = numpy.array(
    [[[2, 0], [0, 1]], [[1, 0], [0, 2]], [[1, 0], [0, 1]]]
)
SETS: Dict[str, int] = {
    "train20": 20,
    "train200": 200,
    "train2000": 2000,
    "validate": 10000,
}


def plot_q_1(
    reg_terms: numpy.ndarray, errors: numpy.ndarray, ml_error: float
) -> axes.Axes:
    """Generate plot for question 1"""
    axs: axes.Axes = seaborn.lineplot(x=reg_terms, y=errors, legend="brief")
    axs.set_ylabel("Mean-squared error")
    axs.set_xlabel("Regularization term")
    pyplot.axhline(ml_error, linestyle="--", color="orange", label="ML Estimator")
    axs.legend(["MAP", "ML"])
    axs.set_xscale("log")
    return axs


def run_q_1():
    """Run question 1"""
    frame = question_1.to_frame()
    train = frame[frame["type"] == "train"]
    valid = frame[frame["type"] == "validate"]
    xtrain: numpy.ndarray = numpy.array(train[["x_1", "x_2"]])
    ytrain: numpy.ndarray = train.y.values
    xval: numpy.ndarray = numpy.array(valid[["x_1", "x_2"]])
    yval: numpy.ndarray = valid.y.values
    errors: numpy.ndarray = numpy.array(list())
    regs = numpy.logspace(-4, 4, 1000)
    design_mat = question_1.simple_design_mat(xtrain)
    with tqdm(total=len(regs)) as pbar:
        for reg in regs:
            map_params = question_1.map_estimate(design_mat, ytrain, reg)
            errors = numpy.append(
                errors, question_1.get_validate_error(map_params, xval, yval)
            )
            pbar.update(1)

    ml_params = question_1.ml_estimate(design_mat, ytrain)
    ml_error = question_1.get_validate_error(ml_params, xval, yval)
    plot_q_1(regs, errors, ml_error)
    pyplot.show()


def like_ratios(xvecs: numpy.ndarray) -> numpy.ndarray:
    """Likelihood ratios for optimal classifier given the question 2 pdf"""
    px_0 = 0.5 * (
        stats.multivariate_normal(MEANS[0], COVS[0]).pdf(xvecs)
        + stats.multivariate_normal(MEANS[1], COVS[1]).pdf(xvecs)
    )
    px_1 = stats.multivariate_normal(MEANS[2], COVS[2]).pdf(xvecs)
    return numpy.array(px_1 / px_0)


def _get_roc(
    arr: numpy.ndarray, labels: numpy.ndarray, priors: numpy.ndarray
) -> numpy.ndarray:
    thresh: float = arr[-1]
    likes: numpy.ndarray = arr[:-1]
    guesses: numpy.ndarray = numpy.where(likes > thresh, 1.0, 0.0)
    l1_idxs = numpy.where(labels == 1.0)[0]
    l0_idxs = numpy.where(labels == 0.0)[0]
    d1_l1 = numpy.where(guesses[l1_idxs] == 1.0)[0]
    d0_l1 = numpy.where(guesses[l1_idxs] == 0.0)[0]
    d1_l0 = numpy.where(guesses[l0_idxs] == 1.0)[0]
    false_pos = d1_l0.shape[0] / l0_idxs.shape[0]
    false_neg = d0_l1.shape[0] / l1_idxs.shape[0]
    error = false_pos * priors[0] + false_neg * priors[1]
    roc = numpy.array(
        [
            false_pos,
            false_neg,
            error,
            d1_l1.shape[0] / l1_idxs.shape[0],
        ]
    )
    return roc


def run_class(
    likes: numpy.ndarray, threshes: numpy.ndarray, labels: numpy.ndarray
) -> pandas.DataFrame:
    """Run the optimal classifier"""
    per_thresh = numpy.repeat(numpy.array([likes]), threshes.shape[0], axis=0)
    with_thresh = numpy.hstack((per_thresh, threshes.reshape((threshes.shape[0], 1))))
    good_and_bad = numpy.apply_along_axis(
        _get_roc, 1, with_thresh, labels, numpy.array([0.65, 0.35])
    )
    frame = pandas.DataFrame(
        numpy.hstack((good_and_bad, threshes.reshape((threshes.shape[0], 1)))),
        columns=["prob_d1l0", "prob_d0l1", "perror", "prob_d1l1", "thresh"],
    )
    return frame


def run_q_2_p_1(xvecs: numpy.ndarray, labels: numpy.ndarray):
    """Run the optimal classifier"""
    likes = like_ratios(xvecs)
    ideal = (
        numpy.where(labels == 0.0)[0].shape[0] / numpy.where(labels == 1.0)[0].shape[0]
    )
    threshes = numpy.sort(numpy.append(numpy.linspace(0, 100, num=9999), ideal))
    roc = run_class(likes, threshes, labels)
    axs: axes.Axes = seaborn.lineplot(data=roc, x="prob_d1l0", y="prob_d1l1")
    theo_pnt = roc.loc[roc["thresh"] == ideal]
    ideal_pnt = roc.loc[roc["perror"] == roc["perror"].min()]
    seaborn.scatterplot(
        data=ideal_pnt, x="prob_d1l0", y="prob_d1l1", ax=axs, color="orange"
    )
    seaborn.scatterplot(
        data=theo_pnt, x="prob_d1l0", y="prob_d1l1", ax=axs, color="green"
    )
    print(ideal_pnt)
    print(theo_pnt)
    axs.set_xlabel("False positive")
    axs.set_ylabel("True positive")
    pyplot.show()


def q_2_p_2_weights(
    frame: pandas.DataFrame, phy_type: str = "lin"
) -> Dict[str, Optional[numpy.ndarray]]:  # pylint: disable=unsubscriptable-object
    """Retrieve all combinations of weights"""
    weights: Dict[
        str, Optional[numpy.ndarray]  # pylint: disable=unsubscriptable-object
    ] = {
        "train20": None,
        "train200": None,
        "train2000": None,
    }
    for set_name in weights:
        xtrain: numpy.ndarray = frame.loc[
            frame["set"] == set_name, ["x_1", "x_2"]
        ].values
        if phy_type == "lin":
            phis = numpy.vstack(
                (numpy.array([[1] * xtrain.shape[0]]), xtrain.transpose())
            )
        elif phy_type == "quad":
            phis = PolynomialFeatures(2).fit_transform(xtrain).transpose()
        else:
            raise ValueError("unknown phi type")
        labels = frame.loc[frame["set"] == set_name, "label"].values
        weights[set_name] = question_2.get_weights(
            labels,
            phis,
            learn_rate=0.00001,
        )

    return weights


def run_q_2_p_2(frame: pandas.DataFrame):
    """Run part two for question 2"""
    xval: numpy.ndarray = frame.loc[frame["set"] == "validate", ["x_1", "x_2"]].values
    yval: numpy.ndarray = frame.loc[frame["set"] == "validate", "label"].values
    lin_phis = numpy.vstack((numpy.array([[1] * xval.shape[0]]), xval.transpose()))
    lin_weights = q_2_p_2_weights(frame, "lin")
    quad_phis = PolynomialFeatures(2).fit_transform(xval).transpose()
    quad_weights = q_2_p_2_weights(frame, "quad")
    print(
        question_2.get_error(
            numpy.array(quad_weights["train20"]), xval, yval, quad_phis
        )
    )
    print(
        question_2.get_error(numpy.array(lin_weights["train20"]), xval, yval, lin_phis)
    )
    print(
        question_2.get_error(
            numpy.array(quad_weights["train200"]), xval, yval, quad_phis
        )
    )
    print(
        question_2.get_error(numpy.array(lin_weights["train200"]), xval, yval, lin_phis)
    )
    print(
        question_2.get_error(
            numpy.array(quad_weights["train2000"]), xval, yval, quad_phis
        )
    )
    print(
        question_2.get_error(
            numpy.array(lin_weights["train2000"]), xval, yval, lin_phis
        )
    )


def run_q_2():
    """Run question 2"""
    frame = question_2.get_data(PRIORS, MEANS, COVS, SETS)
    xval: numpy.ndarray = frame.loc[frame["set"] == "validate", ["x_1", "x_2"]].values
    yval: numpy.ndarray = frame.loc[frame["set"] == "validate", "label"].values
    run_q_2_p_1(xval, yval)
    run_q_2_p_2(frame)


run_q_2()
