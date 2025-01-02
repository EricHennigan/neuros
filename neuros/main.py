import logging
import time
from brainflow.board_shim import BoardIds
from neuros.data_streamer import WindowConfig, create_board_stream, stream_windows

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Demonstrate EEG data streaming with synthetic board.

    This example shows basic usage of the data streaming components:
    - Creates a 550ms window configuration with 225ms overlap
    - Connects to BrainFlow's synthetic board
    - Streams windowed data from first 3 EEG channels
    - Processes windows until interrupted

    The synthetic board provides test signals:
    - Channel 1: 5 Hz sine wave (below alpha band)
    - Channel 2: 10 Hz sine wave (within alpha band)
    - Channel 3: 15 Hz sine wave (above alpha band)

    Press Ctrl+C to stop the stream.

    Raises:
        KeyboardInterrupt: When user stops the program
        Exception: For other unexpected errors during execution
    """
    # Configure window parameters based on synthetic board characteristics
    config = WindowConfig(window_ms=550.0, overlap_ms=225.0)

    try:
        # Initialize board and start streaming
        with create_board_stream(board_id=BoardIds.SYNTHETIC_BOARD) as board:
            sample_rate = board.get_sampling_rate(board_id=board.board_id)
            logger.info(f"Board ready - sample rate: {sample_rate} Hz")
            logger.info("Press Ctrl+C to stop...")

            # Process streaming windows
            for window in stream_windows(board, config):
                # Extract first three channels (known test signals)
                window = window[:3]
                logger.debug(f"Window shape: {window.shape}")

                # Prevent CPU overload in this example
                time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    except Exception as e:
        logger.error(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
