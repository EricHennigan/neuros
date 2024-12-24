import logging
import time
from dataclasses import dataclass
from typing import Iterator, Optional
import numpy as np
from contextlib import contextmanager
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("main.py")


@dataclass(frozen=True)
class WindowConfig:
    """Configuration for data windowing"""
    window_ms: float  # Window size in milliseconds
    overlap_ms: float = 0.0  # Overlap between windows

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.window_ms <= 0:
            raise ValueError("Window size must be positive")
        if self.overlap_ms >= self.window_ms:
            raise ValueError("Overlap must be less than window size")
        if self.overlap_ms < 0:
            raise ValueError("Overlap must be non-negative")

    def to_samples(self, sample_rate: int) -> tuple[int, int]:
        """Convert time-based config to sample counts"""
        window_samples = int(sample_rate * self.window_ms / 1000)
        overlap_samples = int(sample_rate * self.overlap_ms / 1000)
        if overlap_samples >= window_samples:
            raise ValueError("Overlap must be less than window size")
        return window_samples, overlap_samples


@contextmanager
def board_stream(board_id: int = BoardIds.SYNTHETIC_BOARD,
                 params: Optional[BrainFlowInputParams] = None) -> BoardShim:
    """
    Create and manage a board connection.

    Args:
        board_id: BrainFlow board identifier
        params: Optional board parameters

    Yields:
        Connected BoardShim instance

    Raises:
        BrainFlowError: If board initialization or streaming fails
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
    """
    Generate windows of EEG data from board stream.

    This generator yields windows of data continuously from the board stream.
    Each window contains data from all EEG channels and can overlap with previous windows.

    Args:
        board: Connected and streaming BoardShim instance
        config: Window configuration for size and overlap

    Yields:
        Windows of data as numpy arrays with shape (channels, samples)
    """
    # Convert ms-based config to samples
    sample_rate = board.get_sampling_rate(board_id=board.board_id)
    window_samples, overlap_samples = config.to_samples(sample_rate)
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


def main() -> None:
    """Example usage with enhanced visualization"""
    config = WindowConfig(window_ms=550.0, overlap_ms=225.0)

    try:
        with board_stream(board_id=BoardIds.SYNTHETIC_BOARD) as board:
            sample_rate = board.get_sampling_rate(board_id=board.board_id)
            logger.info(f"Board ready - sample rate: {sample_rate} Hz")
            logger.info("Press Ctrl+C to stop...")

            for window in stream_windows(board, config):
                # Get data just for first 3 channels
                window = window[:3]
                logger.debug(f"Window shape: {window.shape}")

                # Small delay to prevent CPU overload
                time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    except Exception as e:
        logger.error(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
