"""Run assignment 3"""
#!/usr/bin/env python3

from typing import Dict, List

import numpy
import pandas
from intro_to_ml_2021.asg_3 import question_1
from sklearn import model_selection, neural_network
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import pyplot
import seaborn
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
        single_label_data = frame.loc[frame["label"] == label]
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
            train_frame = frame.loc[frame["set_name"] == set_name]
            xvals = train_frame[["x0", "x1", "x2"]].values
            targets = train_frame["label"]
            for length in LENGTH_TESTS:
                score_frame = score_frame.append(
                    {
                        "set_name": set_name,
                        "layer_length": length,
                        "score_means": numpy.mean(
                            cross_validation(xvals, targets, length)
                        ),
                    },
                    ignore_index=True,
                )
                pbar.update(1)
    return score_frame


def cross_validation(
    xvals: numpy.ndarray, targets: numpy.ndarray, length: int
) -> numpy.ndarray:
    """Run cross validation"""
    mlp = neural_network.MLPClassifier(
        hidden_layer_sizes=(length,),
        activation="logistic",
        max_iter=10000,
        learning_rate_init=0.00004,
    )
    return numpy.array(
        model_selection.cross_val_score(mlp, xvals, targets, cv=10, n_jobs=-1)
    )


run_question_1()
