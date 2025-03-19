# standard imports
import os
from typing import NoReturn

# package imports
import dill
import numpy as np
import pandas as pd
from sklearn.svm import SVC as SVC_sk

# local imports
from ..utils import logger
from .classifier import Classifier


class SVC(Classifier):

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict with SVC
        """
        return self.kwargs["model"].predict(X)


def build_classifier_SVC(
    data_dir: str,
    kernel: str = "rbf",
    C: float = 1.0,
    debug: bool = True,
) -> NoReturn:
    """
    Build support vector classifier for data at the specified folder.
    Saves the classifier as a dill pickle file to the data folder.

    Parameters
    ----------
    data_dir: str
        folder with data to train classifier. Must contain `bss_svd.csv`

    Returns
    -------
    None
    """
    data_path = os.path.join(data_dir, "bss_svd.csv")

    if debug:
        logger.info(f"Training SVC on {data_path}...")

    # load data
    df = pd.read_csv(data_path, index_col=0)
    X = df[df.columns[4:]].values
    y = df["label"].values

    # fit model
    model = SVC_sk(kernel=kernel, C=C)
    model.fit(X=X, y=y)

    # convert to internal abstraction
    model = SVC(model=model)

    # save model as pickle
    model_path = os.path.join(data_dir, "model_svc.pkl")

    if debug:
        logger.info(f"Saving SVC to {model_path}...")

    with open(model_path, "wb") as file:
        dill.dump(model, file)
