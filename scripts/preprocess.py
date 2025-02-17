# standard imports
import argparse
import os
from typing import Tuple

# local imports
from fm_processor import preprocess_data


def read_input() -> Tuple[str, str]:
    """
    Read the category and node type from the command line arguments.
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
        "output_dir",
        type=str,
        help="directory to save output data and SVD pickle to.",
    )

    args = parser.parse_args()

    return args.data_dir, args.output_dir


if __name__ == "__main__":
    # Read input arguments
    data_dir, output_dir = read_input()

    # make output folder
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # preprocess data
    preprocess_data(data_dir, output_dir)
