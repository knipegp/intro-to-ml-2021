"""Functions for question 2"""
#!/usr/bin/env python3


import multiprocessing
import re
from pathlib import Path
from typing import List

from matplotlib import pyplot, axes
import numpy
import pandas
import seaborn
from matplotlib import axes
from PIL import Image
from sklearn import mixture, preprocessing


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

    pixel_vecs = numpy.hstack((unrolled, pixel_vals))
    norm_vecs = numpy.array(preprocessing.normalize(pixel_vecs))
    return pandas.DataFrame(
        numpy.hstack(
            (
                norm_vecs,
                unrolled,
            )
        ),
        columns=[f"x{idx}" for idx in range(norm_vecs.shape[1])] + ["xcoord", "ycoord"],
    )


def _get_features(frame: pandas.DataFrame) -> List[str]:
    col_names = list(frame.columns)
    features: List[str] = list()
    feature_pattern = re.compile(r"x\d")
    for name in col_names:
        match = feature_pattern.match(str(name))
        if match is not None:
            features.append(match.group(0))
    return features


def gmm_2_component(frame: pandas.DataFrame):
    """Train a two component GMM for the given data"""
    features = _get_features(frame)
    gmm = mixture.GaussianMixture(n_components=2)
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
    gmm.fit(frame[features])
    score = gmm.bic(frame[features])
    return score


def bic_search(frame: pandas.DataFrame) -> pandas.DataFrame:
    """Find the best number of components"""
    features = _get_features(frame)
    comps = numpy.array(numpy.arange(2, 20))
    with multiprocessing.Pool() as pool:
        bic_scores = pool.starmap(_work, [(c, frame, features) for c in comps])
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
