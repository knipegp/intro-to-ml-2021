#  pylint: disable=missing-docstring
#!/usr/bin/env python3

from intro_to_ml_2021.asg_3 import question_1
import numpy


def test_cube_verts():
    verts = question_1.get_cube_verts((0, 0, 0), 2)
    assert (-1.0, -1.0, -1.0) in verts


def test_rand_cov():
    cov: numpy.array = question_1.get_rand_cov(3, (-1, 1), (0.5, 1.4))
    assert cov.shape == (3, 3)


def test_rand_mean():
    mean: numpy.array = question_1.get_random_mean()
    assert len(mean) == 3


# def test_choices():
#     choice_cnts = question_1.choices(1000, [0.25, 0.25, 0.25, 0.25])
#     assert sum(choice_cnts) == 1000
#     assert len(choice_cnts) == 4
