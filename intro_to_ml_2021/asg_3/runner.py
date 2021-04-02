"""Run assignment 3"""
#!/usr/bin/env python3

from typing import Dict, List

import numpy
import pandas
from intro_to_ml_2021.asg_3 import question_1
from sklearn import model_selection, neural_network

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
LENGTH_TESTS = list(range(3, 16))


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
    lengths = get_length_scores(frame)
    print(lengths)


def get_length_scores(frame: pandas.DataFrame) -> Dict[str, int]:
    all_sets = numpy.unique(frame.set_name.values)
    lengths: Dict[str, int] = dict()
    for set_name in all_sets:
        if set_name == "test":
            continue
        train_frame = frame.loc[frame["set_name"] == set_name]
        xvals = train_frame[["x0", "x1", "x2"]].values
        targets = train_frame["label"]
        score_totals: List[float] = list()
        for length in LENGTH_TESTS:
            score_totals.append(sum(cross_validation(xvals, targets, length)))
        len_idx = numpy.array(score_totals).argmax()
        lengths[set_name] = LENGTH_TESTS[len_idx]
    return lengths


def cross_validation(
    xvals: numpy.ndarray, targets: numpy.ndarray, length: int
) -> numpy.ndarray:
    """Run cross validation"""
    mlp = neural_network.MLPClassifier(
        hidden_layer_sizes=(length,), activation="logistic", max_iter=500
    )
    return model_selection.cross_val_score(mlp, xvals, targets, cv=10, n_jobs=-1)


run_question_1()
