"""Functions for question 2"""
#!/usr/bin/env python3


from pathlib import Path
import re
from typing import List

import numpy
import pandas
from PIL import Image
from sklearn import preprocessing, mixture


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


def gmm_2_component(frame: pandas.DataFrame):
    """Train a two component GMM for the given data"""
    col_names = frame.columns
    features: List[str] = list()
    feature_pattern = re.compile("x\d")
    for name in col_names:
        match = feature_pattern.match(name)
        if match is not None:
            features.append(match.group[0])
    print(features)
    # mixture.GaussianMixture(n_components=2).fit()


frame = load_image(Path("~/Downloads/3096_color.jpg").expanduser())
gmm_2_component(frame)
