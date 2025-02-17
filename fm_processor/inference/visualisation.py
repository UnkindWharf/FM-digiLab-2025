# standard imports
from typing import Union, Optional

# package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local imports
from .loading import load_model, load_svd


def plot_decision_boundary(
    model_dir: str,
    model_type: str = "svc",
    data: Optional[Union[str, np.ndarray]] = None,
    output_dir: Optional[str] = None,
):
    """
    Plot the decision boundary of a classifier on a grid
    """
    raise NotImplementedError("TODO")
