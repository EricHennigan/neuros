import logging
import time
from brainflow.board_shim import BoardIds
from neuros.data_streamer import WindowConfig, create_board_stream, stream_windows
from neuros.data_processor import process_window
from tone_generator import ToneGenerator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main() -> None:
    """
    Main function for EEG neurofeedback using Brainflow's synthetic board
    for the data source, and FluidSynth for audio output.
    This version: Uses synthetic board's first channel (5Hz sine wave) for testing.
    """
    # Configure window parameters based on synthetic board characteristics
    config = WindowConfig(window_ms=550.0, overlap_ms=225.0)

    tone = ToneGenerator()

    # Audio feedback parameters
    base_amplitude = 0.3  # Minimum amplitude
    amplitude_boost = 2  # Additional amplitude range


    try:
        # Initialize board and start streaming
        with create_board_stream(board_id=BoardIds.SYNTHETIC_BOARD) as board:
            sample_rate = board.get_sampling_rate(board_id=board.board_id)
            logger.info(f"Board ready - sample rate: {sample_rate} Hz")
            logger.info("Press Ctrl+C to stop...")

            # Process streaming windows
            for window in stream_windows(board, config):
                # Get first channel data
                channel_alpha = window[0:1]  # Just first channel

                # Process window and get alpha ratio
                ratios = process_window(channel_alpha, sample_rate)
                channel_1_ratio = ratios[0]

                # Convert ratio to amplitude
                amplitude = min(base_amplitude + (channel_1_ratio * amplitude_boost), 1.0)

                # Update tone amplitude
                tone.set_amplitude(amplitude)

                logger.debug(f"Alpha ratio: {channel_1_ratio:.3f}, Amplitude: {amplitude:.3f}")

                # Sleep for a while
                time.sleep(config.window_ms / 1000)

    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    except Exception as e:
        logger.error(f"\nError: {e}")
        raise
    finally:
        logger.info("Cleaning up...")
        tone.cleanup()


if __name__ == "__main__":
    main()
