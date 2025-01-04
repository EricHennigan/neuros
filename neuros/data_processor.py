import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations


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


def compute_alpha_ratio(data: np.ndarray, sampling_rate: int) -> float:
    """
    Compute the alpha/total power ratio from EEG data.

    Args:
        data: 1D numpy array of samples.
        sampling_rate: Sampling rate in Hz.

    Returns:
        Alpha/total power ratio as a float.
    """
    alpha_power = extract_band_power(data, sampling_rate, 8, 13)
    total_power = extract_band_power(data, sampling_rate, 1, 50)
    return alpha_power / (total_power + 1e-10)


def process_channel(data: np.ndarray, sampling_rate: int) -> float:
    """
    Process a single channel of EEG data to extract the alpha/total power ratio.

    Args:
        data: 1D numpy array of samples.
        sampling_rate: Sampling rate in Hz.

    Returns:
        Alpha/total power ratio as a float.
    """
    return compute_alpha_ratio(data, sampling_rate)


def process_window(window: np.ndarray, sampling_rate: int) -> list[float]:
    """
    Process a window of multi-channel EEG data.

    Args:
        window: 2D numpy array (channels, samples).
        sampling_rate: Sampling rate in Hz.

    Returns:
        List of alpha/total power ratios, one per channel.
    """
    return [process_channel(window[i], sampling_rate) for i in range(window.shape[0])]


def map_to_midi(alpha_ratio: float) -> int:
    """
    Map alpha/total power ratio (0-1) to MIDI velocity (0-127).

    Args:
        alpha_ratio: Alpha/total power ratio as a float.

    Returns:
        MIDI velocity as an integer.
    """
    return int(np.clip(alpha_ratio * 127, 0, 127))
