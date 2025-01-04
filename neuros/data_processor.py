import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from typing import NamedTuple, Dict


def extract_band_power(data: np.ndarray, sampling_rate: int,
                       low_freq: float, high_freq: float) -> float:
    """
    Extract band power from EEG data.

    Args:
        data: 1D numpy array of samples.
        sampling_rate: Sampling rate in Hz.
        low_freq: Lower frequency bound.
        high_freq: Upper frequency bound.

    Returns:
        Band power as a float.
    """

    filtered = data.copy()
    DataFilter.detrend(filtered, DetrendOperations.CONSTANT.value)
    DataFilter.perform_bandpass(
        filtered,
        sampling_rate,
        low_freq,
        high_freq,
        4,
        FilterTypes.BUTTERWORTH.value,
        0
    )
    return np.sqrt(np.mean(np.square(filtered)))


def extract_all_bands(data: np.ndarray, sampling_rate: int) -> Dict[str, float]:
    """
    Extract power from all standard EEG frequency bands.

    Args:
        data: 1D numpy array of samples
        sampling_rate: Sampling rate in Hz

    Returns:
        Dictionary of band names to power values
    """
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50),
        'total': (1, 50)
    }

    return {
        name: extract_band_power(data, sampling_rate, low, high)
        for name, (low, high) in bands.items()
    }


def compute_band_ratios(powers: Dict[str, float]) -> Dict[str, float]:
    """
    Compute standard ratios between frequency bands.

    Args:
        powers: Dictionary of band powers from extract_all_bands()

    Returns:
        Dictionary of ratio names to values
    """
    ratios = {
        'alpha_theta': ('alpha', 'theta'),
        'alpha_beta': ('alpha', 'beta'),
        'theta_beta': ('theta', 'beta'),
        'alpha_total': ('alpha', 'total')
    }

    return {
        name: powers[num] / (powers[den] + 1e-10)
        for name, (num, den) in ratios.items()
    }


class PowerMetrics(NamedTuple):
    """Collection of power measurements for a channel"""
    absolute_alpha: float
    alpha_ratio: float  # alpha/total
    alpha_beta_ratio: float


def process_channel(data: np.ndarray, sampling_rate: int) -> PowerMetrics:
    """
    Process a single channel of EEG data to extract power metrics.

    Args:
        data: 1D numpy array of samples
        sampling_rate: Sampling rate in Hz

    Returns:
        PowerMetrics containing absolute and relative measurements
    """
    # Get all band powers
    powers = extract_all_bands(data, sampling_rate)
    ratios = compute_band_ratios(powers)

    return PowerMetrics(
        absolute_alpha=powers['alpha'],
        alpha_ratio=ratios['alpha_total'],
        alpha_beta_ratio=ratios['alpha_beta']
    )


def process_window(window: np.ndarray, sampling_rate: int) -> list[PowerMetrics]:
    """
    Process a window of multi-channel EEG data.

    Args:
        window: 2D numpy array (channels, samples)
        sampling_rate: Sampling rate in Hz

    Returns:
        List of PowerMetrics, one per channel
    """
    return [process_channel(window[i], sampling_rate)
            for i in range(window.shape[0])]
