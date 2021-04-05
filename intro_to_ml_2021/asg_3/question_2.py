"""Functions for question 2"""
#!/usr/bin/env python3

from typing import Tuple

import numpy
import pandas
from intro_to_ml_2021.asg_3 import generateMultiringDataset
from sklearn import svm, model_selection
import seaborn
from matplotlib import axes


def gen_data(train_cnt: int, test_cnt: int, class_cnt: int) -> pandas.DataFrame:
    """Generate training and testing samples"""
    samps, labels = generateMultiringDataset.generateMultiringDataset(
        class_cnt, train_cnt + test_cnt
    )
    frame = pandas.DataFrame(
        numpy.vstack(
            (
                samps,
                labels,
            )
        ).transpose(),
        columns=[f"x{idx}" for idx in range(samps.shape[0])] + ["labels"],
    )
    print(frame)
    set_names = numpy.array(["train"] * train_cnt + ["test"] * test_cnt).reshape(-1, 1)
    frame["set_name"] = set_names
    return frame


def check_duplicate(arr: numpy.ndarray):
    """no dups"""
    uniq = numpy.unique(arr)
    for elem in uniq:
        hits = numpy.where(arr == elem)[0]
        assert len(hits) == 1


def parameter_search_svm(frame: pandas.DataFrame) -> pandas.DataFrame:
    """Search for C and gamma values"""
    # c_space = numpy.around(numpy.logspace(1, 2, 50), decimals=2)
    # gamma_space = numpy.around(numpy.logspace(0, 1, 50), decimals=2)
    c_space = numpy.around(numpy.logspace(-3, 0, 30), decimals=4)
    gamma_space = numpy.around(numpy.logspace(-3, 0, 30), decimals=4)
    check_duplicate(c_space)
    check_duplicate(gamma_space)
    params = {"C": c_space, "gamma": gamma_space}
    svc = svm.SVC()
    search = model_selection.GridSearchCV(svc, params, n_jobs=-1, cv=10, verbose=2)
    train_frame = frame.loc[frame["set_name"] == "train"]
    search.fit(train_frame[["x0", "x1"]].values, train_frame["labels"])
    return pandas.DataFrame(search.cv_results_)


def test_best_svm(
    search_res: pandas.DataFrame,
    data_frame: pandas.DataFrame,
) -> Tuple[float, svm.SVC]:
    """Test the best SVM parameters"""
    best_model = search_res.loc[search_res["rank_test_score"] == 1].head(1)
    param_c = best_model["param_C"].values[0]
    param_gamma = best_model["param_gamma"].values[0]
    print(f"best params C {param_c} gamma {param_gamma}")
    svc = svm.SVC(C=param_c, gamma=param_gamma)
    train = data_frame.loc[data_frame["set_name"] == "train"]
    test = data_frame.loc[data_frame["set_name"] == "test"]
    svc.fit(train[["x0", "x1"]], train["labels"])
    return (svc.score(test[["x0", "x1"]], test["labels"]), svc)


def plot_data_and_dec(ax: axes.Axes, data_frame: pandas.DataFrame, svc: svm.SVC):
    test = data_frame.loc[data_frame["set_name"] == "test"]
    xvals = numpy.around(
        numpy.linspace(test.x0.values.min(), test.x0.values.max(), num=100), decimals=3
    )
    yvals = numpy.around(
        numpy.linspace(test.x1.values.min(), test.x1.values.max(), num=100), decimals=3
    )
    gridx, gridy = numpy.meshgrid(xvals, yvals)
    stack = numpy.hstack((gridx.reshape(-1, 1), gridy.reshape(-1, 1)))
    zvals = svc.decision_function(stack).reshape(len(gridy), len(gridx))
    ax.contour(gridx, gridy, numpy.where(zvals < 0, 1.0, 0.0), levels=1)


def plot_q2_data(frame: pandas.DataFrame) -> axes.Axes:
    """Plot the question 2 samples"""
    return seaborn.scatterplot(data=frame, x="x0", y="x1", hue="labels")


def plot_svm_perf(frame: pandas.DataFrame) -> axes.Axes:
    """Plot the performance of the SVM CV"""
    return seaborn.heatmap(
        frame.pivot(index="param_C", columns="param_gamma", values="mean_test_score")
    )
