# standard imports
from abc import ABC, abstractmethod

# package imports
import numpy as np


class Classifier(ABC):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict label given X
        """
