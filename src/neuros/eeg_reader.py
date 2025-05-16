import logging
import numpy as np
import pandas as pd
import threading
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow import DataFilter, DetrendOperations, FilterTypes
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Iterator, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Band(Enum):
    DELTA = (0.5, 4.0)
    THETA = (4.0, 8.0)
    ALPHA = (8.0, 13.0)
    BETA = (13.0, 30.0)
    GAMMA = (30.0, 100.0)
    ALL = (0.5, 100.0)


@dataclass(frozen=True)
class WindowParams:
    """
    Configuration parameters for the thread that grabs raw data from the BCI board
    """
    window: timedelta = timedelta(seconds=0.5) # Window size
    polling: timedelta = timedelta(milliseconds=50) # Time interval between polls
    stop_event: threading.Event = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        object.__setattr__(self, 'stop_event', threading.Event())
        if self.window <= timedelta(0):
            raise ValueError("Window duration must be positive")
        if self.window <= self.polling:
            raise ValueError("Polling interval must be less than the window duration")
        if self.polling < timedelta(0):
            raise ValueError("Polling duration must be positive")


def raw_data(board: BoardShim, params: WindowParams) -> None:
    """
    Run this function in a thread to poll data from the board.
    """
    num_channels = board.get_num_rows(board.board_id)
    samples_per_sec = board.get_sampling_rate(board.board_id)
    num_samples = int(samples_per_sec * params.window.total_seconds()) + 1

    board._data = np.zeros(shape=(num_samples, num_channels))
    board._data_params = params

    board.prepare_session()
    board.start_stream()
    while not params.stop_event.is_set():
        time.sleep(params.polling.total_seconds())
        data = board.get_board_data()
        data = np.transpose(data)
        board._data = np.append(board._data, data, axis=0)[-num_samples:]

    board.stop_stream()
    board.release_session()


@dataclass(frozen=True)
class WindowConfig:
    """
    Configuration for EEG windowing.
    """
    window_ms: float  # Window size in milliseconds
    overlap_ms: float = 0.0  # Overlap between windows

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.window_ms <= 0:
            raise ValueError("Window size must be positive")
        if self.overlap_ms >= self.window_ms:
            raise ValueError("Overlap must be less than window size")
        if self.overlap_ms < 0:
            raise ValueError("Overlap must be non-negative")

    def convert_to_samples(self, sample_rate: int) -> Tuple[int, int]:
        """Convert time-based configuration to sample counts."""
        window_samples = int(sample_rate * self.window_ms / 1000)
        overlap_samples = int(sample_rate * self.overlap_ms / 1000)
        if overlap_samples >= window_samples:
            raise ValueError("Overlap must be less than window size")
        return window_samples, overlap_samples


@contextmanager
def create_board_stream(board_id: int = BoardIds.SYNTHETIC_BOARD, params: Optional[BrainFlowInputParams] = None) -> Iterator[BoardShim]:
    """Create and manage a BrainFlow board connection."""
    board = BoardShim(board_id, params or BrainFlowInputParams())
    try:
        board.prepare_session()
        board.start_stream()
        yield board
    except Exception as e:
        logger.error(f"Error during board operation: {e}")
        raise
    finally:
        try:
            if board.is_prepared():
                board.stop_stream()
                board.release_session()
        except Exception as e:
            logger.error(f"Error during board cleanup: {e}")


def stream_windows(board: BoardShim, config: WindowConfig) -> Iterator[np.ndarray]:
    """
    Stream EEG windows from a board.

    Args:
        board: BoardShim object.
        config: WindowConfig object.

    Yields:
        2D numpy array of EEG data.
    """
    # Convert ms-based config to samples
    sample_rate = board.get_sampling_rate(board_id=board.board_id)
    window_samples, overlap_samples = config.convert_to_samples(sample_rate)
    needed = window_samples - overlap_samples

    # Get EEG channel information
    eeg_channels = board.get_eeg_channels(board_id=board.board_id)

    # Initialize empty aggregator for all EEG data
    aggregator = np.zeros((len(eeg_channels), 0), dtype=np.float32)

    try:
        while True:
            try:
                # Get a large chunk of data - board will return what it has
                new_data = board.get_current_board_data(1024)
                if new_data.size == 0:
                    continue

                # Extract EEG channels and add to aggregator
                eeg_data = new_data[eeg_channels]
                aggregator = np.concatenate([aggregator, eeg_data], axis=1)

                # Generate windows while we have enough data
                while aggregator.shape[1] >= window_samples:
                    # Create the next window
                    window = aggregator[:, :window_samples].copy()

                    # Remove processed data, keeping overlap if needed
                    aggregator = aggregator[:, needed:]

                    yield window

            except Exception as e:
                logger.error(f"Error getting board data: {e}")
                continue

    except GeneratorExit:
        logger.info("Stopping window streaming")


def compute_power(channel_data: np.ndarray, sampling_rate: int, band: Band) -> np.ndarray:
    """
    Extract band power from EEG data for one or more channels.

    Args:
        channel_data: 2D numpy array of samples. Channels are on first axis.
        sampling_rate: Sampling rate in Hz.
        band: Brain wave band.

    Returns:
        Band power as numpy array (2D input).
    """
    low_freq, high_freq = band.value
    filtered = channel_data.copy()

    # Handle both single channel and multi-channel cases
    if filtered.ndim == 1:
        filtered = filtered.reshape(1, -1)

    # Apply filtering to each channel
    for i in range(filtered.shape[0]):
        DataFilter.detrend(filtered[i], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(
            filtered[i],
            sampling_rate,
            low_freq,
            high_freq,
            4,
            FilterTypes.BUTTERWORTH.value,
            0
        )

    # Compute power across all channels
    powers = np.sqrt(np.mean(np.square(filtered), axis=1))

    return powers
