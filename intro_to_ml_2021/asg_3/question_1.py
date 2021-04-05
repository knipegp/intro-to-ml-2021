"""Functions for question 1"""
#!/usr/bin/env python3

from typing import Dict, List, Tuple, Union

import numpy
import pandas
from numpy import linalg, random
from scipy import stats
import seaborn
from sklearn import model_selection, neural_network
from tqdm import tqdm
from matplotlib import axes


def get_cube_verts(
    center: Tuple[float, float, float], length: float
) -> List[Tuple[float, float, float]]:
    """Return the vertices to the specified cube"""
    center_disp = length / 2
    verts: List[Tuple[float, float, float]] = list()
    for mult_x in [1, -1]:
        for mult_y in [1, -1]:
            for mult_z in [1, -1]:
                verts.append(
                    (
                        center_disp * mult_x + center[0],
                        center_disp * mult_y + center[1],
                        center_disp * mult_z + center[2],
                    )
                )

    return verts


def get_random_mean(
    center: Tuple[float, float, float] = (0, 0, 0), length: float = 2
) -> numpy.ndarray:
    """Generate a random mean vector"""
    verts = get_cube_verts(center, length)
    raw_choices = list(range(len(verts)))
    choice = random.choice(raw_choices, replace=False)
    return numpy.array(verts[choice])


def get_rand_cov(
    size: int, var_range: Tuple[float, float], eigval_range: Tuple[float, float]
) -> numpy.ndarray:
    """Generate a random covariance matrix with eigenvalues within the given interval"""
    while True:  # Forgive me
        # for _ in range(1000):
        raw_cov = (var_range[1] - var_range[0]) * random.random_sample(
            (size, size)
        ) + var_range[0]
        symm = numpy.tril(raw_cov, -1).T
        cov = numpy.tril(raw_cov, 0) + symm
        eigs = linalg.eigvals(cov)
        good_eigs = [False] * len(eigs)
        for idx, eigval in enumerate(eigs):
            if eigval < eigval_range[0] or eigval > eigval_range[1]:
                break
            good_eigs[idx] = True
        if False not in good_eigs:
            return cov


def gen_data(
    means: numpy.ndarray,
    covs: numpy.ndarray,
    priors: List[float],
    name_size: Dict[str, int],
) -> pandas.DataFrame:
    """Generate data sets"""
    choice_per_op = random.choice(
        list(range(len(priors))), sum(name_size.values()), p=priors
    )
    single_choices = numpy.unique(choice_per_op)
    samples: Union[numpy.ndarray, None] = None
    for single_choice in single_choices:
        choice_cnt = numpy.where(choice_per_op == single_choice)[0].shape[0]
        samps_with_class = numpy.hstack(
            (
                stats.multivariate_normal(
                    mean=means[single_choice], cov=covs[single_choice]
                ).rvs(choice_cnt),
                numpy.array([single_choice] * choice_cnt, ndmin=2).T,
            )
        )
        if samples is None:
            samples = samps_with_class
        else:
            samples = numpy.append(samples, samps_with_class, axis=0)
    raw_sample_sets: List[str] = list()
    for set_name, set_size in name_size.items():
        raw_sample_sets += [set_name] * set_size
    sample_sets = numpy.array(raw_sample_sets).reshape((len(raw_sample_sets), 1))
    random.shuffle(samples)
    frame: pandas.DataFrame = pandas.DataFrame(
        samples, columns=["x0", "x1", "x2", "label"]
    )
    frame["set_name"] = sample_sets
    return frame


def get_best_length(lens: numpy.ndarray, scores: numpy.ndarray) -> int:
    """Return the best hidden layer length given the performance stats"""
    coef = numpy.polyfit(lens, scores, 3)
    fit = numpy.polyval(coef, lens)
    delta = numpy.diff(fit)
    last_inc_idx = numpy.where(delta <= 0)[0][0]
    return lens[:last_inc_idx].max()


def train_and_score(
    train_x: numpy.ndarray,
    train_label: numpy.ndarray,
    test_x: numpy.ndarray,
    test_label: numpy.ndarray,
    layer_len: int,
) -> float:
    """Train and test a model"""
    mlp = neural_network.MLPClassifier(
        hidden_layer_sizes=(layer_len,),
        activation="logistic",
        max_iter=10000,
        learning_rate_init=0.00004,
    )
    mlp.fit(train_x, train_label)
    return mlp.score(test_x, test_label)


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


def _guess_class(
    samp: numpy.ndarray,
    dists: List[Tuple[float, stats._multivariate.multivariate_normal_frozen]],
) -> int:
    likes: List[float] = list()
    for prior_dist in dists:
        prior = prior_dist[0]
        dist = prior_dist[1]
        likes.append(dist.pdf(samp) * prior)
    pdf = sum(likes)
    return int(numpy.argmin(numpy.array([1 - like / pdf for like in likes])))


def evaluate_optimal(
    data: pandas.DataFrame,
    means: numpy.ndarray,
    covs: numpy.ndarray,
    priors: numpy.ndarray,
) -> float:
    """Evaluate the optimal classifier. Return the accuracy rate."""
    dists: List[Tuple[float, stats._multivariate.multivariate_normal_frozen]] = list()
    for idx, prior in enumerate(priors):
        mean = means[idx]
        cov = covs[idx]
        dist = stats.multivariate_normal(mean=mean, cov=cov)
        dists.append((prior, dist))
    test = data.loc[data["set_name"] == "test"]
    samps = test[["x0", "x1", "x2"]].values
    guesses: List[float] = [0.0] * samps.shape[0]
    for idx, samp in enumerate(samps):
        guesses[idx] = float(_guess_class(samp, dists))
    data.loc[data["set_name"] == "test", "guess"] = guesses
    right_cnt = len(numpy.where(test["label"] == guesses)[0])
    return right_cnt / samps.shape[0]


def get_length_scores(
    frame: pandas.DataFrame, length_tests: List[int]
) -> pandas.DataFrame:
    """Get the inferred hidden layer length for each classifier"""
    all_sets = numpy.unique(frame.set_name.values)
    score_frame: pandas.DataFrame = pandas.DataFrame(
        columns=["set_name", "layer_length", "score_means"]
    )
    with tqdm(total=(len(all_sets) * len(length_tests))) as pbar:
        for set_name in all_sets:
            if set_name == "test":
                continue
            train_frame: pandas.DataFrame = frame.loc[frame["set_name"] == set_name]
            xvals: numpy.ndarray = numpy.array(train_frame[["x0", "x1", "x2"]].values)
            targets = numpy.array(train_frame["label"].values)
            for length in length_tests:
                score_frame = score_frame.append(
                    {
                        "set_name": set_name,
                        "layer_length": length,
                        "score_means": numpy.mean(
                            numpy.array(cross_validation(xvals, targets, length))
                        ),
                    },
                    ignore_index=True,
                )
                pbar.update(1)
    return score_frame


def plot_scores_lengths(scores: pandas.DataFrame) -> axes.Axes:
    """Plot mlp scores"""
    plot: axes.Axes = seaborn.lineplot(
        data=scores, x="layer_length", y="score_means", hue="set_name"
    )
    return plot


def plot_acc_train_len(
    scores: List[float], train_size: List[int], opt: float
) -> axes.Axes:
    """Plot the best scores for the train data size"""
    plot: axes.Axes = seaborn.lineplot(
        data={"accuracy": scores, "train dataset size": train_size},
        y="accuracy",
        x="train dataset size",
    )
    plot.set_xscale("log")
    plot.axhline(opt, linestyle="--")
    return plot
