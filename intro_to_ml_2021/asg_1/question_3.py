"""Support functions for question 3"""
#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, List

import numpy
from numpy import linalg
import pandas
from scipy import stats


def get_info(label_file: Path) -> Dict[int, str]:
    """Retrieve info from dataset files"""
    raw_labels = numpy.loadtxt(label_file.expanduser(), dtype=str)
    return {int(elem[0]): elem[1] for elem in raw_labels}


def get_mean_vectors(
    labels: numpy.ndarray,
    frame: pandas.DataFrame,
    alpha: float,
) -> List[stats._multivariate.multivariate_normal_frozen]:
    """Get a mean vector for each label"""
    dists = list()
    for lab in numpy.unique(labels):
        idxs = numpy.where(labels == lab)[0]
        mean = frame.loc[idxs].mean().values
        cov = frame.loc[idxs].cov().values
        try:
            dists.append(stats.multivariate_normal(mean=mean, cov=cov))
        except linalg.LinAlgError:
            lamda = alpha * numpy.trace(cov) / linalg.matrix_rank(cov)
            cov = cov + lamda * numpy.identity(cov.shape[0])
            dists.append(stats.multivariate_normal(mean=mean, cov=cov))

    return dists
