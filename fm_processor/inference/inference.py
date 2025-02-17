# standard imports
import os
from typing import Union, Literal

# package imports
import numpy as np
import pandas as pd

# local imports
from .loading import load_model, load_svd
from ..preprocess import wavelet_denoise


def predict(
    data: str,
    model_dir: str,
    model_type: str = "svc",
    denoise_wavelet: str = "sym4",
    denoise_level: int = 12,
    denoise_method: Literal["universal", "energy"] = "universal",
    denoise_thresholding: str = "soft",
    denoise_energy: float = 0.98,
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
    # extract temperature
    temperature = float(data[-8:-4]) / 100

    # load data
    data = np.sum(np.load(data)[1:, :], axis=0)

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
    df_train = pd.read_csv(os.path.join(model_dir, "bss.csv"), index_col=0)
    # get closest clean signal in terms of temperature
    idx = (
        (temperature - df_train.loc[df_train.label == 0].temperature).abs().argsort()[0]
    )
    # extract signal
    baseline_sample = np.hstack(
        df_train.loc[df_train.label == 0].iloc[idx][df_train.columns[5:]].values
    )

    # baseline subtraction
    data = (data - baseline_sample)[np.newaxis, :]

    # load model
    svd = load_svd(model_dir)
    model = load_model(model_dir, model_type)

    # inference
    pred = model.predict(svd.transform(data))

    return pred
