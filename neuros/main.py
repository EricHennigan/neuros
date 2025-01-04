import logging
import time
from brainflow.board_shim import BoardIds
from neuros.data_streamer import WindowConfig, create_board_stream, stream_windows
from neuros.data_processor import process_window
from tone_generator import ToneGenerator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def main() -> None:
    """Demonstrate EEG data streaming with a simple tone generator."""
    config = WindowConfig(window_ms=550.0, overlap_ms=225.0)

    # Initialize ToneGenerator
    tone_generator = ToneGenerator()

    try:
        with create_board_stream(board_id=BoardIds.SYNTHETIC_BOARD) as board:
            sample_rate = board.get_sampling_rate(board_id=board.board_id)
            logger.info(f"Board ready - sample rate: {sample_rate} Hz")
            logger.info("Press Ctrl+C to stop...")

            tone_generator.play()  # Start playing the tone

            for window in stream_windows(board, config):
                # Extract alpha/total ratio for channel 1
                ratios = process_window(window, sample_rate)
                channel_1_ratio = ratios[0]  # Assuming channel 1 is index 0

                # Update the tone volume (Only channel 1 for now)
                tone_generator.set_volume(channel_1_ratio)

                time.sleep(0.1)  # Adjust for desired responsiveness

    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    except Exception as e:
        logger.error(f"\nError: {e}")
        raise
    finally:
        tone_generator.stop()  # Stop tone and clean up


if __name__ == "__main__":
    main()
