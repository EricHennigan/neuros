import logging
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple
import numpy as np
from contextlib import contextmanager
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowConfig:
    """Configuration for EEG data windowing.

    This class handles configuration for windowing continuous EEG data,
    supporting overlapping windows for analysis.

    Attributes:
        window_ms: Window size in milliseconds.
        overlap_ms: Overlap between consecutive windows in milliseconds.
            Defaults to 0.0 (no overlap).

    Raises:
        ValueError: If window_ms is not positive, overlap_ms is negative,
            or overlap_ms is greater than or equal to window_ms.
    """
    window_ms: float
    overlap_ms: float = 0.0

    def __post_init__(self) -> None:
        """Validate window configuration parameters.

        Raises:
            ValueError: If window parameters are invalid.
        """
        if self.window_ms <= 0:
            raise ValueError("Window size must be positive")
        if self.overlap_ms >= self.window_ms:
            raise ValueError("Overlap must be less than window size")
        if self.overlap_ms < 0:
            raise ValueError("Overlap must be non-negative")

    def convert_to_samples(self, sample_rate: int) -> Tuple[int, int]:
        """Convert time-based configuration to sample counts.

        Args:
            sample_rate: Sampling rate in Hz.

        Returns:
            Tuple containing (window_samples, overlap_samples).

        Raises:
            ValueError: If resulting overlap is greater than or equal to window size.
        """
        window_samples = int(sample_rate * self.window_ms / 1000)
        overlap_samples = int(sample_rate * self.overlap_ms / 1000)
        if overlap_samples >= window_samples:
            raise ValueError("Overlap must be less than window size")
        return window_samples, overlap_samples


@contextmanager
def create_board_stream(board_id: int = BoardIds.SYNTHETIC_BOARD,
                       params: Optional[BrainFlowInputParams] = None) -> Iterator[BoardShim]:
    """Create and manage a BrainFlow board connection.

    Context manager that handles board initialization, streaming, and cleanup.

    Args:
        board_id: BrainFlow board identifier. Defaults to synthetic board.
        params: Optional board parameters. If None, uses default parameters.

    Yields:
        Connected BoardShim instance ready for data streaming.

    Raises:
        BrainFlowError: If board initialization or streaming fails.
        Exception: For other unexpected errors during board operation.
    """
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
    """Generate windows of EEG data from a board stream.

    Creates a generator that continuously yields windows of EEG data from
    the board stream. Windows can overlap based on the configuration.

    Args:
        board: Connected and streaming BoardShim instance.
        config: Configuration specifying window size and overlap.

    Yields:
        np.ndarray: Windows of data with shape (channels, samples).
            Each window contains data from all EEG channels.

    Raises:
        GeneratorExit: When streaming is stopped.
        Exception: For errors during data acquisition.

    Notes:
        - Uses a 1024-sample buffer for data acquisition
        - Automatically handles partial data chunks
        - Thread-safe for concurrent access
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
