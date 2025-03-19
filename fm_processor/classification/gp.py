# standard imports
import os
from typing import NoReturn

# package imports
import dill
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier

# local imports
from ..utils import logger
from .classifier import Classifier


class GP(Classifier):

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict with GP
        """
        return self.kwargs["model"].predict(X)


def build_classifier_GP(
    data_dir: str,
    training_iterations=200,
    debug: bool = True,
) -> NoReturn:
    """
    Build Bernoulli Gaussian Process classifier for data at the specified folder.
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
        logger.info(f"Training GP on {data_path}...")

    # load data
    df = pd.read_csv(data_path, index_col=0)
    X = df[df.columns[4:]].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    # build model
    model = GaussianProcessClassifier()

    # fit model
    model.fit(X=X, y=y)

    # convert to internal abstraction
    gp = GP(model=model)

    # save model as pickle
    model_path = os.path.join(data_dir, "model_gp.pkl")

    if debug:
        logger.info(f"Saving GP to {model_path}...")

    with open(model_path, "wb") as file:
        dill.dump(gp, file)
