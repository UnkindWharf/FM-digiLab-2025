# standard imports
import argparse
from typing import Tuple

# local imports
from fm_processor import build_classifier


def read_input() -> Tuple[str, str]:
    """
    Read script inputs
    """

    parser = argparse.ArgumentParser(
        description="Preprocess raw signal data in a folder"
    )

    parser.add_argument(
        "data_dir",
        type=str,
        help="directory to read raw .npy files.",
    )

    parser.add_argument(
        "model_type",
        type=str,
        default="svc",
        help="""Model type to fit. Options: 
            "svc": Support Vector Classifier
            "gp": Bernoulli Gaussian Process
            """,
    )

    args = parser.parse_args()

    return args.data_dir, args.model_type


if __name__ == "__main__":
    # Read input arguments
    data_dir, model_type = read_input()

    # preprocess data
    build_classifier(data_dir, model_type)
