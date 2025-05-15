import logging
import time

import numpy as np
from brainflow.board_shim import BoardIds
from neuros.eeg_reader import WindowConfig, create_board_stream, stream_windows, Band, compute_power
from neuros.tone_generator import ToneGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Example usage demonstrating window streaming with synthetic board"""
    config = WindowConfig(window_ms=550.0, overlap_ms=225.0)
    max_alpha_ratios = np.zeros(8)  # Track max for each channel
    windows_to_skip = 500  # Number of windows to skip, to allow for stabilization
    cpu_delay = 0.005  # Small delay between iterations to prevent CPU overload
    tone = ToneGenerator()

    try:
        with create_board_stream(board_id=BoardIds.SYNTHETIC_BOARD) as board:
            sample_rate = board.get_sampling_rate(board_id=board.board_id)
            logger.info(f"Board ready - sample rate: {sample_rate} Hz")
            logger.info("Press Ctrl+C to stop...")

            window_iterator = stream_windows(board, config)

            # Skip the first windows_to_skip windows
            for _ in range(windows_to_skip):
                next(window_iterator)

            for window in window_iterator:
                channels = window[:8]
                alpha_powers = compute_power(channels, sample_rate, Band.ALPHA)
                total_powers = compute_power(channels, sample_rate, Band.ALL)
                alpha_ratios = alpha_powers / (total_powers + 1e-10)

                # Update maximum ratios per channel
                new_maxes = alpha_ratios > max_alpha_ratios
                if np.any(new_maxes):
                    max_alpha_ratios[new_maxes] = alpha_ratios[new_maxes]
                    logger.info(f"New max alpha ratios: {max_alpha_ratios}")

                # Normalize each channel by its own maximum
                normalized_alpha_ratios = alpha_ratios / (max_alpha_ratios + 1e-10)
                logger.info(f"Normalized alpha ratios: {normalized_alpha_ratios}")
                tone.set_velocity(normalized_alpha_ratios)

                # Small delay to prevent CPU overload
                time.sleep(cpu_delay)

    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    except Exception as e:
        logger.error(f"\nError: {e}")
        raise
    finally:
        tone.cleanup()


if __name__ == "__main__":
    main()
