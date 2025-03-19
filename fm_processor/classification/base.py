# local imports
from .svc import build_classifier_SVC
from .gp import build_classifier_GP


def build_classifier(
    data_dir: str,
    model_type: str = "svc",
    **kwargs,
):
    """
    Convenience function for building classifiers on signal data. Model
    will be saved to data directory.

    Parameters
    ----------
    data_dir: str,
        directory with model pickle file
    model_type: str
        Model type to load. Options:
            "svc": Support Vector Classifier
    """
    if model_type == "svc":
        build_classifier_SVC(data_dir=data_dir, **kwargs)
    elif model_type == "gp":
        build_classifier_GP(data_dir=data_dir, **kwargs)
