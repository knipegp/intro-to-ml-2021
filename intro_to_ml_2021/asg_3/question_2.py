"""Functions for question 2"""
#!/usr/bin/env python3

from typing import Tuple

import numpy
import pandas
from intro_to_ml_2021.asg_3 import generateMultiringDataset
from sklearn import svm, model_selection


def gen_data(train_cnt: int, test_cnt: int, class_cnt: int) -> pandas.DataFrame:
    """Generate training and testing samples"""
    samps, labels = generateMultiringDataset.generateMultiringDataset(
        class_cnt, train_cnt + test_cnt
    )
    frame = pandas.DataFrame(
        numpy.hstack(
            (
                samps.reshape(-1, class_cnt),
                labels.reshape(-1, 1),
            )
        ),
        columns=[f"x{idx}" for idx in range(samps.shape[0])] + ["labels"],
    )
    set_names = numpy.array(["train"] * train_cnt + ["test"] * test_cnt).reshape(-1, 1)
    frame["set_name"] = set_names
    return frame


def parameter_search_svm(frame: pandas.DataFrame) -> pandas.DataFrame:
    """Search for C and gamma values"""
    params = {"C": numpy.logspace(-1, 3, 15), "gamma": numpy.logspace(-1, 3, 15)}
    svc = svm.SVC()
    search = model_selection.GridSearchCV(svc, params, n_jobs=-1, cv=10, verbose=2)
    train_frame = frame.loc[frame["set_name"] == "train"]
    search.fit(train_frame[["x0", "x1"]].values, train_frame["labels"])
    return pandas.DataFrame(search.cv_results_)
