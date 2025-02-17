# standard imports
import argparse
import os
from typing import Tuple

# package imports
import numpy as np

# local imports
from fm_processor import predict


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
        help="path to read test data .npy files.",
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="directory to load SVD and model pickle files",
    )

    parser.add_argument(
        "output_path",
        type=str,
        help="path to save output CSV. if not provided, results will be saved to model_dir",
        default=None,
    )

    args = parser.parse_args()

    return args.data_path, args.model_dir, args.output_path


if __name__ == "__main__":
    # Read input arguments
    data_path, model_dir, output_path = read_input()

    # preprocess data
    pred = predict(data_path, model_dir)

    # save results
    if output_path is None:
        output_path = os.path.join(model_dir, "inference.npy")

    np.save(output_path, pred)
