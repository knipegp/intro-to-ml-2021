"""Support functions for question 3"""
#!/usr/bin/env python3
from pathlib import Path
from typing import Dict

import numpy


def get_info(label_file: Path) -> Dict[int, str]:
    """Retrieve info from dataset files"""
    raw_labels = numpy.loadtxt(label_file.expanduser(), dtype=str)
    return {int(elem[0]): elem[1] for elem in raw_labels}
