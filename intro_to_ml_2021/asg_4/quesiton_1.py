"""Functions for question 1"""
#!/usr/bin/env python3

import numpy
import pandas

from intro_to_ml_2021.asg_2 import hw2q1

TRAIN_SET = 1000
TEST_SET = 10000


def generate_data(train_set: int, test_set: int) -> pandas.DataFrame:
    """Generate gaussian data from assignment 2"""
    raw_data, raw_labels = hw2q1.generateData(train_set + test_set)
    frame = pandas.DataFrame(
        numpy.hstack(
            (
                raw_data.transpose(),
                raw_labels.reshape(-1, 1),
            )
        ),
        columns=[f"x{idx}" for idx in range(raw_data.shape[0])] + ["label"],
    )
    frame["set_name"] = ["train"] * train_set + ["test"] * test_set
    return frame
