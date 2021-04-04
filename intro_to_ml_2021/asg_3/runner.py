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
# LENGTH_TESTS: List[int] = list(range(9, 20, 3))


def run_question_1():
    """Run question 1 functions"""
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
    score_means = get_length_scores(frame)
    score_means.to_csv("./scores.csv")
    print(score_means)


def plot_data(frame: pandas.DataFrame):
    """Plot generated 3d data"""
    fig = pyplot.figure()
    plot: Axes3D = fig.add_subplot(111, projection="3d")
    for label, color in zip(numpy.unique(frame.label.values), ["r", "g", "b"]):
        single_label_data: pandas.DataFrame = frame.loc[frame["label"] == label]
        plot.scatter(
            single_label_data.x0, single_label_data.x1, single_label_data.x2, c=color
        )
    pyplot.show()


def get_length_scores(frame: pandas.DataFrame) -> pandas.DataFrame:
    """Get the inferred hidden layer length for each classifier"""
    all_sets = numpy.unique(frame.set_name.values)
    score_frame: pandas.DataFrame = pandas.DataFrame(
        columns=["set_name", "layer_length", "score_means"]
    )
    with tqdm(total=(len(all_sets) * len(LENGTH_TESTS))) as pbar:
        for set_name in all_sets:
            if set_name == "test":
                continue
            train_frame: pandas.DataFrame = frame.loc[frame["set_name"] == set_name]
            xvals: numpy.ndarray = train_frame[["x0", "x1", "x2"]].values
            targets = train_frame["label"]
            for length in LENGTH_TESTS:
                score_frame = score_frame.append(
                    {
                        "set_name": set_name,
                        "layer_length": length,
                        "score_means": numpy.mean(
                            question_1.cross_validation(xvals, targets, length)
                        ),
                    },
                    ignore_index=True,
                )
                pbar.update(1)
    return score_frame


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
            train_set: pandas.DataFrame = sample_frame.loc[
                sample_frame["set_name"] == set_name
            ]
            scores.append(
                question_1.train_and_score(
                    train_set[["x0", "x1", "x2"]],
                    train_set["label"],
                    test[["x0", "x1", "x2"]],
                    test["label"],
                    layer_len,
                )
            )
    return scores


# print(
#     train_and_score_all(
#         pandas.read_csv("../../scores.csv"), pandas.read_csv("../../frame.csv")
#     )
# )
frame = question_2.gen_data(1000, 10000, 2)
frame.to_csv("svm_data.csv")
results = question_2.parameter_search_svm(frame)
results.to_csv("svm_results.csv")
