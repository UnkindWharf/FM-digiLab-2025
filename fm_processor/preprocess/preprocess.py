# standard imports
import os
from typing import List, Dict, Tuple, Literal, NoReturn, Any

# package imports
import dill
import numpy as np
import pandas as pd
from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD

# local imports
from ..utils import logger
from .denoise import WaveletDenoising


def preprocess_data(
    data_dir: str,
    output_dir: str,
    denoise_wavelet: str = "sym4",
    denoise_level: int = 12,
    denoise_method: Literal["universal", "energy"] = "universal",
    denoise_thresholding: str = "soft",
    denoise_energy: float = 0.98,
    svd_retained_variance: float = 0.999,
    debug: bool = True,
) -> NoReturn:
    """
    Load data, apply Wavelet denoising, and truncated SVD. Saves postprocessed data and
    SVD to directory of choice

    Parameters
    ----------
    data_dir: str
        directory to load the .NPY raw data. All files starting with "clean_" are
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
    if debug:
        logger.info(f"Reading raw data from {data_dir} ...")
    filenames, labels, temperatures, data = load_data(data_dir=data_dir)

    # apply Wavelet denoising
    if debug:
        logger.info("Applying Wavelet denoising...")
    for i in range(len(data)):
        data[i] = wavelet_denoise(
            data[i],
            wavelet=denoise_wavelet,
            level=denoise_level,
            method=denoise_method,
            thresholding=denoise_thresholding,
            energy=denoise_energy,
        )

    # save raw data before baseline subtraction
    df_params = pd.DataFrame(
        {
            "filename": filenames,
            "label": labels,
            "temperature": temperatures,
        }
    )
    df_signal = pd.DataFrame(data)
    df_data = pd.concat([df_params, df_signal], axis=1)
    df_data.to_csv(os.path.join(output_dir, "data.csv"))

    # apply baseline subtraction to all defects and clean with 1 degree either side
    if debug:
        logger.info("Applying baseline subtraction...")
    bss_signal = []  # nodal summed signals
    bss_baseline = []  # nodal summed baseline
    bss_labels = []  # defect=1, clean=0
    bss_temperatures = []  # signal temp
    bss_temp_diff = []  # signal temp - clean temp
    bss_data = []  # signal - clean

    for i in range(len(data)):
        for j in range(len(data)):
            if (
                (i != j)
                and (labels[j] == 0)
                and (temperatures[j] - 1 < temperatures[j] < temperatures[j] + 1)
            ):
                bss_signal.append(filenames[i])
                bss_baseline.append(filenames[j])
                bss_labels.append(labels[i])
                bss_temperatures.append(temperatures[i])
                bss_temp_diff.append(temperatures[i] - temperatures[j])
                bss_data.append(data[i] - data[j])

    # apply truncated SVD to data
    if debug:
        logger.info("Applying SVD...")
    bss_data = np.vstack(bss_data)
    bss_data_svd, svd = truncated_svd(bss_data, svd_retained_variance)

    # save bss data
    if debug:
        logger.info(f"Saving preprocessed data to {output_dir} ...")
    df_params = pd.DataFrame(
        {
            "filename_signal": bss_signal,
            "filename_baseline": bss_baseline,
            "label": bss_labels,
            "temperature_diff": bss_temp_diff,
            "temperature": bss_temperatures,
        }
    )
    df_signal = pd.DataFrame(bss_data)
    df_data = pd.concat([df_params, df_signal], axis=1)
    df_data.to_csv(os.path.join(output_dir, "bss.csv"))

    # save svd data
    df_signal = pd.DataFrame(bss_data_svd)
    df_data = pd.concat([df_params, df_signal], axis=1)
    df_data.to_csv(os.path.join(output_dir, "bss_svd.csv"))

    # save svd object
    with open(os.path.join(output_dir, "svd.pkl"), "wb") as file:
        dill.dump(svd, file)

    if debug:
        logger.info("Data preprocessing finished.")


def load_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load raw data to memory. All files starting with "clean_" are
    labelled clean signals, everything else is labelled defect. All files expected
    to end with _xxxx.npy, where xxxx denotes the temperature. Each individual
    signal file is assumed to have 2 columns: timestamp and signal. The
    timestamp is ignored.

    Parameters
    ----------
    data_dirs: str
        directory to load the .NPY raw data.

    Return
    ------
    Tuple[List, List, List, List]
        Tuple with lists of filename, label, temperature, data
    """

    # loading utility
    def load_dir(dir):
        filenames = []
        labels = []
        temperatures = []
        data = []

        for root, dirs, files in os.walk(dir):
            for file in files:
                # extract raw signal
                signal = np.load(os.path.join(root, file))[1:, :]
                # sum all nodal values
                signal = np.sum(signal, axis=0)
                # extract temperature information
                temp = float(file[-8:-4]) / 100
                # extract label
                label = 0 if "clean" in file else 1
                # add to list
                filenames.append(os.path.join(root, file))
                labels.append(label)
                temperatures.append(temp)
                data.append(signal)

        return filenames, labels, temperatures, data

    # process provided directory
    filenames, labels, temperatures, data = load_dir(data_dir)

    return filenames, labels, temperatures, data


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
