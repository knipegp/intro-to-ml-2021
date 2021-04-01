"""Run assignment 3"""
#!/usr/bin/env python3

from intro_to_ml_2021.asg_3 import question_1

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


def run_question_1():
    """Run question 1 functions"""
    means: List[numpy.ndarray] = list()
    covs: List[numpy.ndarray] = list()
    for _ in range(len(PRIORS)):
        means.append(question_1.get_random_mean())
        covs.append(question_1.get_rand_cov(DIMS, (-1, 1), (0.5, 1.4)))
    question_1.gen_data(means, covs, PRIORS, SET_NAME_SIZES)


run_question_1()
