import logging
import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("processing.py")

def extract_band_power(data: np.ndarray, sampling_rate: int,
                       low_freq: float, high_freq: float) -> float:
    """Extract power in a specific frequency band from signal."""
    filtered = data.copy()

    # Remove DC offset
    DataFilter.detrend(filtered, DetrendOperations.CONSTANT.value)

    # Apply bandpass filter
    DataFilter.perform_bandpass(
        filtered,
        sampling_rate,
        low_freq, high_freq,
        4,  # Order
        FilterTypes.BUTTERWORTH.value,
        0
    )

    # Compute RMS power
    power = np.sqrt(np.mean(np.square(filtered)))
    return power


def process_window(window: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Process a window of EEG data to extract normalized alpha power.
    Returns array of alpha powers, one per channel.
    """
    num_channels = window.shape[0]
    alpha_powers = np.zeros(num_channels)

    for i in range(num_channels):
        channel_data = window[i]

        # Get powers in different bands
        alpha_power = extract_band_power(channel_data, sampling_rate, 8, 13)
        total_power = extract_band_power(channel_data, sampling_rate, 1, 40)  # Full range
        # Optionally get other bands for ratios:
        # theta_power = extract_band_power(channel_data, sampling_rate, 4, 8)
        # beta_power = extract_band_power(channel_data, sampling_rate, 13, 32)

        # Normalize alpha power by total power to get relative strength
        alpha_powers[i] = alpha_power / (total_power + 1e-10)  # Avoid division by zero

        # Alternative: alpha/theta ratio
        # alpha_powers[i] = alpha_power / (theta_power + 1e-10)

    return alpha_powers


def extract_all_bands(data: np.ndarray, sampling_rate: int) -> dict[str, float]:
    """Extract power from all traditional EEG bands."""
    return {
        'delta': extract_band_power(data, sampling_rate, 1, 4),
        'theta': extract_band_power(data, sampling_rate, 4, 8),
        'alpha': extract_band_power(data, sampling_rate, 8, 13),
        'beta': extract_band_power(data, sampling_rate, 13, 30),
        'gamma': extract_band_power(data, sampling_rate, 30, 50)
    }


def compute_band_ratios(data: np.ndarray, sampling_rate: int) -> dict[str, float]:
    """Compute common band ratios used in EEG analysis."""
    bands = extract_all_bands(data, sampling_rate)
    return {
        'alpha/theta': bands['alpha'] / (bands['theta'] + 1e-10),
        'theta/beta': bands['theta'] / (bands['beta'] + 1e-10),
        'alpha/beta': bands['alpha'] / (bands['beta'] + 1e-10),
        'alpha/delta': bands['alpha'] / (bands['delta'] + 1e-10)
    }


def get_spectral_features(data: np.ndarray, sampling_rate: int, window_ms: float = 500.0) -> dict[str, float]:
    """Calculate spectral features of the signal."""
    from brainflow.data_filter import DataFilter

    # Get PSD using Welch's method
    psd_data = data.copy()
    DataFilter.detrend(psd_data, DetrendOperations.CONSTANT.value)

    # Find nearest power of 2 that's less than our data length
    nfft = 64  # Start with a reasonable power of 2
    while nfft * 2 <= len(data):
        nfft *= 2

    overlap = nfft // 2  # 50% overlap

    # Get PSD
    try:
        psd = DataFilter.get_psd_welch(
            psd_data,
            nfft,
            overlap,
            sampling_rate,
            nfft  # Using same value for window size
        )

        # Extract features
        return {
            'peak_frequency': np.argmax(psd[1]) * psd[0][1],
            'mean_power': np.mean(psd[1]),
            'median_power': np.median(psd[1]),
            'peak_power': np.max(psd[1]),
            'power_variance': np.var(psd[1]),
            'psd': psd[1]  # Full PSD for visualization
        }
    except Exception as e:
        logger.error(f"Error in spectral analysis: {e}")
        # Return empty dictionary if analysis fails
        return {
            'peak_frequency': 0,
            'mean_power': 0,
            'median_power': 0,
            'peak_power': 0,
            'power_variance': 0,
            'psd': np.zeros(nfft // 2 + 1)  # Empty PSD array
        }
