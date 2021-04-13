"""Run assignment 4"""
#!/usr/bin/env python3

from pathlib import Path

import pandas
from intro_to_ml_2021.asg_3 import question_1 as asg3q1
from intro_to_ml_2021.asg_4 import quesiton_1, question_2
from matplotlib import pyplot
import seaborn


def generate_question_1():
    """Generate data for question 1"""
    frame = quesiton_1.generate_data(quesiton_1.TRAIN_SET, quesiton_1.TEST_SET)
    frame.to_csv("./data.csv")


def cross_validate_layer_lens():
    """Cross validate the MLP to find the optimal hidden layer length"""
    frame = pandas.DataFrame(pandas.read_csv("./data.csv", index_col=0))
    scores = asg3q1.get_length_scores(
        frame, list(range(3, 101)), 10000, 0.00004, "neg_mean_squared_error", "relu"
    )
    scores.to_csv("./scores.csv")


def gmm_2_images():
    for path in [Path("./3096_color.jpg"), Path("./42049_color.jpg")]:
        frame = question_2.load_image(path.expanduser())
        question_2.gmm_2_component(frame)
        question_2.plot_gmm_over_image(path.expanduser(), frame)
        pyplot.savefig(Path(path.stem + "_plot.jpg"))
        pyplot.close()


def gmm_comp_search():
    for path in [Path("./3096_color.jpg"), Path("./42049_color.jpg")]:
        frame = question_2.load_image(path.expanduser())
        bic_scores = question_2.bic_search(frame)
        bic_scores.to_csv(Path(path.stem + "_scores.csv"))
        seaborn.scatterplot(data=bic_scores, x="component_count", y="bic_score")
        pyplot.savefig(Path(path.stem + "_bic.jpg"))
        pyplot.close()
