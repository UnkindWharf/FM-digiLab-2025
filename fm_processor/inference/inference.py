# standard imports
import os
from typing import List, Literal

# package imports
import numpy as np
import pandas as pd

# local imports
from .loading import load_model, load_svd
from ..preprocess import wavelet_denoise


def predict_file(data_path: str, model_dir: str, model_type: str = "svc", **kwargs):
    """
    Run inference on one test data file

    Parameters
    ----------
    data: str
        a string to a .npy file with the test data.
    model_dir: str
        directory with model pickle file
    model_type: str
        Model type to load. Options:
            "svc": Support Vector Classifier
    """
    # extract temperature
    temperature = float(data_path[-8:-4]) / 100

    # load data
    data = np.sum(np.load(data_path)[1:, :], axis=0)

    return predict(
        data=data,
        temperature=temperature,
        model_dir=model_dir,
        model_type=model_type,
        **kwargs
    )


def predict_array(
    data: List[np.ndarray],
    temperatures: List[float],
    model_dir: str,
    model_type: str = "svc",
    **kwargs
):
    """
    Run inference on test data array

    Parameters
    ----------
    data: numpy.ndarray
        2D numpy array with test data
    temperatures: List[float]
        list of temperatures for each row of data matrix
    model_dir: str
        directory with model pickle file
    model_type: str
        Model type to load. Options:
            "svc": Support Vector Classifier
    """
    preds = []

    for i in range(data.shape[0]):
        preds.append(
            predict(
                data=data[i],
                temperature=temperatures[i],
                model_dir=model_dir,
                model_type=model_type,
                **kwargs
            )
        )


def predict(
    data: np.ndarray,
    temperature: float,
    model_dir: str,
    model_type: str = "svc",
    denoise_wavelet: str = "sym4",
    denoise_level: int = 12,
    denoise_method: Literal["universal", "energy"] = "universal",
    denoise_thresholding: str = "soft",
    denoise_energy: float = 0.98,
):
    """
    Predict one sample of data in numpy format
    """
    assert data.ndim == 1, "Data array must be 1D"

    # denoise data
    data = wavelet_denoise(
        data,
        wavelet=denoise_wavelet,
        level=denoise_level,
        method=denoise_method,
        thresholding=denoise_thresholding,
        energy=denoise_energy,
    )

    # read training data as baseline database
    df_train = pd.read_csv(os.path.join(model_dir, "data.csv"), index_col=0)
    # get closest clean signal in terms of temperature
    idx = (
        (temperature - df_train.loc[df_train.label == 0].temperature)
        .abs()
        .argsort()
        .values[0]
    )
    # extract signal
    baseline_sample = np.hstack(
        df_train.loc[df_train.label == 0].iloc[idx][df_train.columns[3:]].values
    )

    # baseline subtraction
    data = (data - baseline_sample)[np.newaxis, :]

    # load model
    svd = load_svd(model_dir)
    model = load_model(model_dir, model_type)

    # transform signal
    data = svd.transform(data)

    # prepend temperature column
    data = np.hstack([[[temperature]], data])

    # inference
    pred = model.predict(data)

    return pred[0]
