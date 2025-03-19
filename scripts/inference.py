# standard imports
import argparse
import os
from typing import Tuple, Optional

# package imports
import numpy as np

# local imports
from fm_processor import predict_file


def read_input() -> Tuple[str, str, str]:
    """
    Read script inputs
    """

    parser = argparse.ArgumentParser(
        description="Preprocess raw signal data in a folder"
    )

    parser.add_argument(
        "data_path",
        type=str,
        help="path to read test data .npy file.",
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="directory to load SVD and model pickle files",
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

    return args.data_path, args.model_dir, args.model_type


if __name__ == "__main__":
    # Read input arguments
    data_path, model_dir, model_type = read_input()

    # preprocess data
    pred = predict_file(data_path, model_dir, model_type)

    print(f"Classification result: {pred}")
