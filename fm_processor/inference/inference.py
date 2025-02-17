# standard imports
from typing import Union

# package imports
import numpy as np

# local imports
from .loading import load_model, load_svd


def predict(
    data: Union[str, np.ndarray],
    model_dir: str,
    model_type: str = "svc",
):
    """
    Run inference on test data

    Parameters
    ----------
    data: Union[str, numpy.ndarray]
        a string to a .npy file with the test data, or a nunmpy array.
    model_dir: str
        directory with model pickle file
    model_type: str
        Model type to load. Options:
            "svc": Support Vector Classifier
    """
    # load data
    if isinstance(data, str):
        data = np.load(data)

    # sanity check
    assert data.ndim == 2, f"Test data must be 2D numpy array. Got {data.ndim}D array."

    # load model
    svd = load_svd(model_dir)
    model = load_model(model_dir, model_type)

    # inference
    pred = model.predict(svd.transform(data))

    return pred
