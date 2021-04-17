"""Run assignment 4"""
#!/usr/bin/env python3

from pathlib import Path

import numpy
import pandas
from intro_to_ml_2021.asg_3 import question_1 as asg3q1
from intro_to_ml_2021.asg_4 import question_1, question_2
from matplotlib import pyplot
import seaborn


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
    data = pandas.DataFrame(pandas.read_csv("./data.csv", index_col=0))
    scores = pandas.DataFrame(pandas.read_csv("./scores.csv", index_col=0))
    question_1.calculate_mses(data, scores)


def plot_gmm(path: Path, comp_cnt: int):
    frame = question_2.load_image(path.expanduser())
    question_2.gmm_2_component(frame, comp_cnt)
    question_2.plot_gmm_over_image(path.expanduser(), frame)
    pyplot.savefig(Path(path.stem + f"_{comp_cnt}" + "_plot.jpg"))
    pyplot.close()


def gmm_2_images():
    for path in [Path("./3096_color.jpg"), Path("./42049_color.jpg")]:
        plot_gmm(path, 2)


def gmm_comp_search():
    for path in [Path("./3096_color.jpg"), Path("./42049_color.jpg")]:
        frame = question_2.load_image(path.expanduser())
        comps = numpy.array(numpy.arange(2, 20))
        bic_scores = question_2.bic_search(frame, comps)
        bic_scores.to_csv(Path(path.stem + "_scores.csv"))


def plot_comp_search():
    for path in [Path("./3096_color_scores.csv"), Path("./42049_color_scores.csv")]:
        frame = pandas.read_csv(path, index_col=0)
        seaborn.scatterplot(data=frame, x="component_count", y="bic_score")
        pyplot.savefig(Path(path.stem + "comp_search.jpg"))
        pyplot.close()


def plot_q_1(frame: pandas.DataFrame):
    fig = pyplot.figure()
    plot: Axes3D = fig.add_subplot(111, projection="3d")
    for label, color in zip(numpy.unique(frame.label.values), ["r", "g", "b"]):
        single_label_data: pandas.DataFrame = frame.loc[frame["label"] == label]
        plot.scatter(
            single_label_data.x0, single_label_data.x1, single_label_data.x2, c=color
        )
    pyplot.show()
    # pyplot.savefig("./q13d.png", format="png")


# f = pandas.read_csv("./data.csv", index_col=0)
# plot_q_1(f)
calc_mses()
# gmm_2_images()
# gmm_comp_search()
# plot_comp_search()
# plot_gmm(Path("./3096_color.jpg"), 5)
# plot_gmm(Path("./42049_color.jpg"), 11)
