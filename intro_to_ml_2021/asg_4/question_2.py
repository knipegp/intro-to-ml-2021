"""Functions for question 2"""
#!/usr/bin/env python3

import math
import re
from pathlib import Path
from typing import List

import numpy
from numpy import random
import pandas
from matplotlib import axes, pyplot
from PIL import Image
from sklearn import mixture
from tqdm import tqdm


def _normalize(arr: numpy.ndarray) -> numpy.ndarray:
    arrmin = arr.min()
    arrmax = arr.max() - arrmin
    arr = arr - arrmin
    arr = arr / arrmax
    return arr


def load_image(path: Path) -> pandas.DataFrame:
    """Load the image into a dataframe"""
    with Image.open(path) as image:
        xlen, ylen = image.size
        pixels = image.load()

    grid = numpy.meshgrid(numpy.arange(xlen), numpy.arange(ylen))
    unrolled = numpy.hstack(
        (
            grid[0].flatten().reshape(-1, 1),
            grid[1].flatten().reshape(-1, 1),
        )
    )

    pixel_vals = numpy.array([pixels[coord[0], coord[1]] for coord in unrolled])
    norm = numpy.hstack(
        (
            _normalize(grid[0].flatten()).reshape(-1, 1),
            _normalize(grid[1].flatten()).reshape(-1, 1),
        )
    )
    for col in range(pixel_vals.shape[1]):
        norm = numpy.hstack(
            (
                norm,
                _normalize(
                    pixel_vals[:, col].reshape(
                        -1,
                    )
                ).reshape(-1, 1),
            )
        )

    frame = pandas.DataFrame(
        numpy.hstack(
            (
                norm,
                unrolled,
            )
        ),
        columns=[f"x{idx}" for idx in range(norm.shape[1])] + ["xcoord", "ycoord"],
    )
    frame["ycoord"] = numpy.flip(frame["ycoord"].values)
    frame["x1"] = numpy.flip(frame["x1"].values)
    return frame


def _get_features(frame: pandas.DataFrame) -> List[str]:
    col_names = list(frame.columns)
    features: List[str] = list()
    feature_pattern = re.compile(r"x\d")
    for name in col_names:
        match = feature_pattern.match(str(name))
        if match is not None:
            features.append(match.group(0))
    return features


def gmm_2_component(frame: pandas.DataFrame, comp_cnt: int = 2):
    """Train a two component GMM for the given data"""
    features = _get_features(frame)
    gmm = mixture.GaussianMixture(n_components=comp_cnt)
    frame["label"] = gmm.fit_predict(frame[features]).reshape(-1, 1)


def plot_gmm_over_image(path: Path, frame: pandas.DataFrame) -> axes.Axes:
    """plot the gmm"""
    plot: axes.Axes = pyplot.axes()
    xvals = numpy.array(frame.xcoord.values)
    yvals = numpy.array(frame.ycoord.values)
    xcnt = numpy.unique(xvals).shape[0]
    ycnt = numpy.unique(yvals).shape[0]
    xgrid = xvals.reshape(ycnt, xcnt)
    ygrid = yvals.reshape(ycnt, xcnt)
    labels = numpy.array(frame.label.values).reshape(ycnt, xcnt)
    plot.contour(xgrid, ygrid, labels)
    with Image.open(path) as image:
        plot.imshow(image, extent=[0, image.size[0] - 1, 0, image.size[1] - 1])
    return plot


def _work(
    count: int,
    frame: pandas.DataFrame,
    features: List[str],
) -> float:
    gmm = mixture.GaussianMixture(n_components=count)
    samples = frame[features]
    gmm.fit(samples)
    # all_probs: List[float] = list()
    # for comp_idx, _ in enumerate(gmm.means_):
    #     mean = gmm.means_[comp_idx]
    #     cov = gmm.covariances_[comp_idx]
    #     probs = stats.multivariate_normal.pdf(samples, mean=mean, cov=cov)
    #     all_probs.append(probs.sum())
    # prob = float(numpy.array(all_probs).sum())
    feat_cnt = len(features)
    complexity = (
        (count - 1) + feat_cnt * count + count * feat_cnt * (feat_cnt + 1) / 2
    ) * math.log(feat_cnt * frame.shape[0])
    lower = -2 * gmm.score(samples)
    # print(lower)
    # print(complexity)
    return lower + complexity * 0.001


def bic_search(frame: pandas.DataFrame, comps: numpy.ndarray) -> pandas.DataFrame:
    """Find the best number of components"""
    features = _get_features(frame)
    bic_scores: List[float] = list()
    with tqdm(total=comps.shape[0]) as bar:
        for comp_cnt in comps:
            bic_scores.append(_work(comp_cnt, frame, features))
            bar.update(1)
    scores = pandas.DataFrame(
        numpy.hstack(
            (
                numpy.array(bic_scores).reshape(-1, 1),
                comps.reshape(-1, 1),
            )
        ),
        columns=["bic_score", "component_count"],
    )
    return scores
