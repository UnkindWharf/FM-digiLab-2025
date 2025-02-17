# standard imports
import os
from typing import NoReturn

# package imports
import dill
import pandas as pd
from sklearn.svm import SVC

# local imports
from ..utils import logger


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
    X = df[df.columns[5:]].values
    y = df["label"].values

    # fit model
    model = SVC(kernel=kernel, C=C)
    model.fit(X=X, y=y)

    # save model as pickle
    model_path = os.path.join(data_dir, "model_svc.pkl")

    if debug:
        logger.info(f"Saving SVC to {model_path}...")

    with open(model_path, "wb") as file:
        dill.dump(model, file)
