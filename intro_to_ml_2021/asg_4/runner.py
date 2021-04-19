"""Run assignment 4"""
#!/usr/bin/env python3

from pathlib import Path

import numpy
import pandas
import seaborn
from intro_to_ml_2021.asg_3 import question_1 as asg3q1
from intro_to_ml_2021.asg_4 import question_1, question_2
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import neural_network


def generate_question_1():
    """Generate data for question 1"""
    frame = question_1.generate_data(question_1.TRAIN_SET, question_1.TEST_SET)
    frame.to_csv("./data.csv")


def cross_validate_layer_lens():
    """Cross validate the MLP to find the optimal hidden layer length"""
    frame = pandas.DataFrame(pandas.read_csv("./data.csv", index_col=0))
    scores = asg3q1.get_length_scores(
        frame, list(range(3, 101)), 10000, 0.00004, "neg_mean_squared_error", "relu"
    )
    scores.to_csv("./scores.csv")


def calc_mses():
    """Calculate and print mean-squared errors"""
    data = pandas.DataFrame(pandas.read_csv("./data.csv", index_col=0))
    question_1.calculate_mses(data)


def plot_gmm(path: Path, comp_cnt: int):
    """Plot fit GMM data over photo"""
    frame = question_2.load_image(path.expanduser())
    question_2.gmm_2_component(frame, comp_cnt)
    question_2.plot_gmm_over_image(path.expanduser(), frame)
    pyplot.savefig(Path(path.stem + f"_{comp_cnt}" + "_plot.jpg"))
    pyplot.close()


def gmm_2_images():
    """Plot 2-component GMM"""
    for path in [Path("./3096_color.jpg"), Path("./42049_color.jpg")]:
        plot_gmm(path, 2)


def gmm_comp_search():
    """Find the best number of GMM components"""
    for path in [Path("./3096_color.jpg"), Path("./42049_color.jpg")]:
        frame = question_2.load_image(path.expanduser())
        comps = numpy.array(numpy.arange(2, 20))
        bic_scores = question_2.bic_search(frame, comps)
        bic_scores.to_csv(Path(path.stem + "_scores.csv"))


def plot_comp_search():
    """Plot the BIC scores for each component count"""
    for path in [Path("./3096_color_scores.csv"), Path("./42049_color_scores.csv")]:
        frame = pandas.read_csv(path, index_col=0)
        seaborn.scatterplot(data=frame, x="component_count", y="bic_score")
        pyplot.savefig(Path(path.stem + "comp_search.jpg"))
        pyplot.close()


def plot_q_1():
    """Plot the 3D data and whether it is guessed correctly"""
    frame = pandas.read_csv("./data.csv", index_col=0)
    train = frame.loc[frame["set_name"] == "train"]
    mlp = neural_network.MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        max_iter=10000,
        learning_rate_init=0.00004,
    )
    mlp.fit(train[["x0", "x1", "x2"]], train["label"])
    test = frame.loc[frame["set_name"] == "test"]
    test["guess"] = mlp.predict(test[["x0", "x1", "x2"]])
    fig = pyplot.figure()
    plot: Axes3D = fig.add_subplot(111, projection="3d")
    for label, color in zip(numpy.unique(test.label.values), ["r", "g", "b"]):
        for is_correct, marker in zip([True, False], ["o", "x"]):
            single_label_data: pandas.DataFrame = test.loc[test["label"] == label]
            is_correct_data = single_label_data.loc[
                (single_label_data["guess"] == single_label_data["label"]) == is_correct
            ]
            plot.scatter(
                is_correct_data.x0,
                is_correct_data.x1,
                is_correct_data.x2,
                c=color,
                marker=marker,
            )
    pyplot.show()
