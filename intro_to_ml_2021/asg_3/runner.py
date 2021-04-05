"""Run assignment 3"""
#!/usr/bin/env python3

from typing import List

import numpy
import pandas
from intro_to_ml_2021.asg_3 import question_1, question_2
from matplotlib import pyplot
from mpl_toolkits.mplot3d.axes3d import Axes3D
from tqdm import tqdm

PRIORS = [0.25, 0.25, 0.25, 0.25]
DIMS = 3
SET_NAME_SIZES = {
    "train100": 100,
    "train200": 200,
    "train500": 500,
    "train1000": 1000,
    "train2000": 2000,
    "train5000": 5000,
    "test": 100000,
}
LENGTH_TESTS: List[int] = list(range(1, 101))


def gen_question_1():
    """Gen question 1 data"""
    means: List[numpy.ndarray] = list()
    covs: List[numpy.ndarray] = list()
    for _ in range(len(PRIORS)):
        means.append(question_1.get_random_mean())
        covs.append(question_1.get_rand_cov(DIMS, (-1, 1), (0.5, 1.4)))
    frame = question_1.gen_data(
        numpy.array(means), numpy.array(covs), PRIORS, SET_NAME_SIZES
    )
    numpy.save("./means.npy", numpy.array(means))
    numpy.save("./covs.npy", numpy.array(covs))
    frame.to_csv("./frame.csv")
    score_means = question_1.get_length_scores(frame, LENGTH_TESTS)
    score_means.to_csv("./scores.csv")


def run_question_1_mlp() -> List[float]:
    """Run the question 1 mlp"""
    return train_and_score_all(
        pandas.DataFrame(pandas.read_csv("scores.csv")),
        pandas.DataFrame(pandas.read_csv("frame.csv")),
    )


def plot_data(frame: pandas.DataFrame):
    """Plot generated 3d data"""
    fig = pyplot.figure()
    plot: Axes3D = fig.add_subplot(111, projection="3d")
    for label, color in zip(numpy.unique(frame.label.values), ["r", "g", "b"]):
        single_label_data: pandas.DataFrame = frame.loc[frame["label"] == label]
        plot.scatter(
            single_label_data.x0, single_label_data.x1, single_label_data.x2, c=color
        )
    pyplot.savefig("./q13d.png", format="png")


def train_and_score_all(
    score_frame: pandas.DataFrame, sample_frame: pandas.DataFrame
) -> List[float]:
    """Train and score all classifiers"""
    scores: List[float] = list()
    test: pandas.DataFrame = sample_frame.loc[sample_frame["set_name"] == "test"]
    with tqdm(total=len(SET_NAME_SIZES)) as pbar:
        for set_name in SET_NAME_SIZES:
            pbar.update(1)
            if set_name == "test":
                continue
            scores_set: pandas.DataFrame = score_frame.loc[
                score_frame["set_name"] == set_name
            ]
            layer_len = question_1.get_best_length(
                scores_set.layer_length.values, scores_set.score_means.values
            )
            print(layer_len)
            train_set: pandas.DataFrame = sample_frame.loc[
                sample_frame["set_name"] == set_name
            ]
            scores.append(
                question_1.train_and_score(
                    numpy.array(train_set[["x0", "x1", "x2"]].values),
                    numpy.array(train_set["label"].values),
                    numpy.array(test[["x0", "x1", "x2"]].values),
                    numpy.array(test["label"].values),
                    layer_len,
                )
            )
    return scores


def gen_question_2():
    """Generate question 2 data"""
    frame = question_2.gen_data(1000, 10000, 2)
    frame.to_csv("svm_data_new.csv")
    results = question_2.parameter_search_svm(frame)
    results.to_csv("svm_results_new.csv")


def run_question_2():
    """Run question 2"""
    frame = pandas.DataFrame(pandas.read_csv("svm_data_new.csv", index_col=0))
    ax = question_2.plot_q2_data(frame)
    results = pandas.DataFrame(pandas.read_csv("svm_results_new.csv", index_col=0))
    scores, svc = question_2.test_best_svm(results, frame)
    print(scores)
    question_2.plot_data_and_dec(ax, frame.loc[frame["set_name"] == "test"], svc)
    pyplot.savefig("./svm_test_data.png")
    pyplot.close()
    question_2.plot_svm_perf(results)
    pyplot.savefig("./svm_param_search.png", bbox_inches="tight", pad_inches=0)
    pyplot.close()


def run_question_1_opt() -> float:
    """Run the optimal classifier for question 1"""
    frame = pandas.DataFrame(pandas.read_csv("frame.csv", index_col=0))
    means = numpy.load("./means.npy")
    covs = numpy.load("./covs.npy")
    return question_1.evaluate_optimal(frame, means, covs, numpy.array([0.25] * 4))


def plot_question_1():
    """Generate question 1 plots"""
    frame = pandas.DataFrame(pandas.read_csv("frame.csv", index_col=0))
    scores = pandas.DataFrame(pandas.read_csv("scores.csv", index_col=0))
    plot_data(frame)
    pyplot.close()
    question_1.plot_scores_lengths(scores)
    pyplot.savefig("./q1scores_layer_lengths.png")
    pyplot.close()
    opt = run_question_1_opt()
    print(opt)
    best_scores = run_question_1_mlp()
    print(best_scores)
    question_1.plot_acc_train_len(best_scores, [100, 200, 500, 1000, 2000, 5000], opt)
    pyplot.savefig("./q1scores_data_length.png")
    pyplot.close()
