"""Functions for question 1"""
#!/usr/bin/env python3

import numpy
import pandas
from sklearn import neural_network, metrics, model_selection

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


def calculate_mses(data: pandas.DataFrame, scores: pandas.DataFrame):
    min_mean_idx = scores.score_means.argmax()
    best_score = scores.iloc[min_mean_idx]
    layer_len = best_score.layer_length
    mlp = neural_network.MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        max_iter=10000,
        learning_rate_init=0.00004,
    )
    train = data.loc[data["set_name"] == "train"]
    test = data.loc[data["set_name"] == "test"]
    mlp.fit(train[["x0", "x1", "x2"]], train["label"])
    print(mlp.coefs_[0].shape)
    print(mlp.intercepts_[0].shape)
    x_range = numpy.linspace(data.x0.min(), data.x0.max(), num=100)
    y_range = numpy.linspace(data.x1.min(), data.x1.max(), num=100)
    z_range = numpy.linspace(data.x2.min(), data.x2.max(), num=100)
    x_grid, y_grid, z_grid = numpy.meshgrid(x_range, y_range, z_range)
    all_points = numpy.hstack(
        (
            x_grid.reshape(-1, 1),
            y_grid.reshape(-1, 1),
            z_grid.reshape(-1, 1),
        )
    )
    print(all_points.shape)
    # numpy.polyval(mlp.coefs_[0][0])
    # guesses = mlp.predict(test[["x0", "x1", "x2"]])
    # score = model_selection.cross_val_score(
    #     neural_network.MLPClassifier(
    #         hidden_layer_sizes=(16,),
    #         activation="relu",
    #         max_iter=10000,
    #         learning_rate_init=0.00004,
    #     ),
    #     train[["x0", "x1", "x2"]],
    #     train["label"],
    #     scoring="neg_mean_squared_error",
    #     cv=10,
    #     n_jobs=-1,
    # ).mean()
    # print(-score)
    # print(metrics.mean_squared_error(test["label"], guesses))
