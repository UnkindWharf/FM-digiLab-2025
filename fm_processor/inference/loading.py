# standard imports
import os
from typing import Any

# package imports
import dill


def load_model(
    path: str,
    model_type: str = "svc",
) -> Any:
    """
    Load a classifier model into memory for inference.

    Parameters
    ----------
    path: str
        Path to directory containing the model pickle
    model_type: str, Optional
        Model type to load. Options:
            "svc": Support Vector Classifier
    """
    # load support vector classifier
    if model_type == "svc":
        path = os.path.join(path, "model_svc.pkl")
    elif model_type == "gp":
        path = os.path.join(path, "model_gp.pkl")

    with open(path, "rb") as file:
        obj = dill.load(file)

    return obj


def load_svd(
    path: str,
) -> Any:
    """
    Load a svd object into memory for inference.

    Parameters
    ----------
    path: str
        Path to directory containing the svd pickle
    """
    path = os.path.join(path, "svd.pkl")

    with open(path, "rb") as file:
        obj = dill.load(file)

    return obj
