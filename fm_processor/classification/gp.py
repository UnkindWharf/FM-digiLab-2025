# standard imports
import os
from typing import NoReturn

# package imports
import dill
import numpy as np
import pandas as pd
import torch
from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

# local imports
from ..utils import logger
from .classifier import Classifier


class GP(Classifier):

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict with GP
        """
        X = torch.from_numpy(X.astype(np.float32))

        self.kwargs["likelihood"].eval()
        self.kwargs["model"].eval()

        preds = self.kwargs["likelihood"](self.kwargs["model"](X))

        return preds.mean.detach().numpy()


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

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    # build model
    model = GPClassificationModel(X)
    likelihood = BernoulliLikelihood()

    # fit model
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the number of training datapoints
    mll = VariationalELBO(likelihood, model, y.numel())

    for i in range(training_iterations):
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        output = model(X)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        loss.backward()
        print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iterations, loss.item()))
        optimizer.step()

    model.eval()
    likelihood.eval()

    # convert to internal abstraction
    gp = GP(
        model=model,
        likelihood=likelihood,
    )

    # save model as pickle
    model_path = os.path.join(data_dir, "model_gp.pkl")

    if debug:
        logger.info(f"Saving GP to {model_path}...")

    with open(model_path, "wb") as file:
        dill.dump(gp, file)


class GPClassificationModel(ApproximateGP):
    """
    https://docs.gpytorch.ai/en/v1.13/examples/04_Variational_and_Approximate_GPs/Non_Gaussian_Likelihoods.html
    """

    def __init__(self, train_x):
        """
        Initialise the model and variational distribution
        """
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=1.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = MultivariateNormal(mean_x, covar_x)
        return latent_pred
