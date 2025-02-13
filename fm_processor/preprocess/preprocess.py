# standard imports
import os
from typing import List, Dict, Tuple, Literal, NoReturn, Any

# package imports
import numpy as np
from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD

# local imports
from .denoise import WaveletDenoising


def preprocess_data(
    data_dirs: List[str],
    output_dir: str,
    denoise_wavelet: str = "sym4",
    denoise_level: int = 12,
    denoise_method: Literal["universal", "energy"] = "universal",
    denoise_thresholding: str = "soft",
    denoise_energy: float = 0.98,
    svd_retained_variance: float = 0.999,
) -> NoReturn:
    """
    Load data, apply Wavelet denoising, and truncated SVD. Saves postprocessed data and
    SVD to directory of choice

    Parameters
    ----------
    data_dirs: List[str]
        list of folders to load the .NPY raw data. All files starting with "clean_" are
        labelled clean signals, everything else is labelled defect. All files expected
        to end with _xxxx.npy, where xxxx denotes the temperature.
    output_dir: str
        output directory to write results to. Will write a CSV with the original data,
        a CSV with the decomposed data, and the truncated SVD transformation. The
        transformation is necessary because fast SVD uses random initialisations, so
        the transform must be kept for inverse projection into the original signal
        space.

    Return
    ------
    None
    """
    # load raw data
    data = load_data(data_dirs=data_dirs)

    for sample in data:
        # apply Wavelet denoising
        sample["data"] = wavelet_denoise(
            sample["data"],
            wavelet=denoise_wavelet,
            level=denoise_level,
            method=denoise_method,
            thresholding=denoise_thresholding,
            energy=denoise_energy,
        )

        # apply truncated SVD
        sample["data_svd"], svd = truncated_svd(sample["data"], svd_retained_variance)

    # save processed data TODO


def load_data(data_dirs: List[str]) -> List[Dict[str, Any]]:
    """
    Load raw data to memory. All files starting with "clean_" are
    labelled clean signals, everything else is labelled defect. All files expected
    to end with _xxxx.npy, where xxxx denotes the temperature. Each individual
    signal file is assumed to have 2 columns: timestamp and signal. The
    timestamp is ignored.

    Parameters
    ----------
    data_dirs: List[str]
        list of folders to load the .NPY raw data.

    Return
    ------
    List[Dict[str, np.ndarray]]
        list with dictionaries with keys ["filename", "label", "temperature", "data"]
    """

    # loading utility
    def load_dir(dir):
        results = []

        for root, dirs, files in os.walk(dir):
            for file in files:
                # extract raw signal
                data = np.load(os.path.join(root, file))[1, :]
                # extract temperature information
                temp = float(file[-8:-4]) / 100
                # extract label
                label = 0 if "clean" in file else 1
                # add to list
                results.append(
                    {
                        "filename": file,
                        "label": label,
                        "temperature": temp,
                        "data": data,
                    }
                )

        return results

    # process all provided directories
    all_results = []
    for data_dir in data_dirs:
        all_results = all_results + load_dir(data_dir)

    return all_results


def wavelet_denoise(
    data: np.ndarray,
    wavelet: str = "sym4",
    level: int = 12,
    method: Literal["universal", "energy"] = "universal",
    thresholding: Literal["soft", "hard"] = "soft",
    energy: float = 0.98,
) -> np.ndarray:
    """
    Apply Wavelet denoising with discrete Wavelet decomposition

    Parameters
    ----------
    data: np.ndarray
        numpy vector with raw data.
    wavelet: str, Optional
        the pywavelets string representation of a wavelet suitable
        for discrete Wavelet decomposition. Defaults to sym4
    level: int, Optional
        the pywavelets number of decomposition levels. If this is
        set too high, a pywavelets warning will be printed.
        Defaults to 12 which will work well for FullMatrix signals.
    method: Literal["universal", "energy"], Optional
        the type of method used in fm_processor.preprocess.denoise.WaveletDenoising.
        Universal thresholding should work well in almost all scenarios.
    thresholding: Literal["soft", "hard"], Optional
        whether to apply soft or hard thresholding. Both will set
        all coefficients smaller than the computed threshold to zero.
        For all other coefficients hard threshold will not change,
        soft threshold will interpolate based on distance to threshold.
        For more information on threshold computation see https://github.com/gdetor/wavelet_denoising
    energy: float, Optional
        The percentage of energy [0..1] to keep in the signal for energy
        threshold.
    """
    assert (
        data.ndim == 1
    ), "fm_processor.preprocess.wavelet_denoise() expects data to be 1D"

    wd = WaveletDenoising(
        normalize=False,
        wavelet=wavelet,
        level=level,
        thr_mode=thresholding,
        selected_level=None,
        method=method,
        energy_perc=energy,
    )

    return wd.fit(data)


def truncated_svd(
    data: np.ndarray,
    retained_variance: float = 0.999,
) -> Tuple[np.ndarray, TruncatedSVD]:
    """
    Compute truncated SVD of the data with a desired retained variance ratio
    """
    truncation_length = get_truncation_length(data, retained_variance)

    svd = TruncatedSVD(n_components=truncation_length)
    svd.fit(data)

    # return data and SVD object
    return svd.transform(data), svd


def get_truncation_length(
    data: np.ndarray,
    retained_variance: float,
) -> int:
    """
    Compute truncation length based on desired retained variance
    """
    # decompose with compact SVD
    _, _, vh = svd(data, full_matrices=False)

    # project data along singular vectors
    Y_transformed = np.matmul(data, vh.T)

    # calculate explained variance
    explained_variance = np.var(Y_transformed, axis=0) / np.var(data, axis=0).sum()

    # calculate truncation length
    truncation_length = (
        np.argmax((np.cumsum(explained_variance, axis=0) > retained_variance)) + 1
    )

    return truncation_length
