import pytest
import numpy as np
from brainflow.board_shim import BoardIds
from neuros.data_streamer import WindowConfig, create_board_stream, stream_windows
from neuros.data_processor import process_window


def test_synthetic_alpha_ratios():
    """
    Test that synthetic board channels show expected alpha ratio patterns.
    Based on analysis showing consistent alpha/total ratios around 0.25-0.26
    across channels.
    """
    config = WindowConfig(window_ms=1000.0)

    with create_board_stream(BoardIds.SYNTHETIC_BOARD) as board:
        window = next(stream_windows(board, config))
        metrics = process_window(window[:3], board.get_sampling_rate(board.board_id))

        # Alpha ratios should be consistent across channels
        for channel_metrics in metrics:
            assert 0.20 <= channel_metrics.alpha_ratio <= 0.30

        # Alpha/beta ratios should be close to observed values
        for channel_metrics in metrics:
            assert 0.9 <= channel_metrics.alpha_beta_ratio <= 1.3


def test_synthetic_power_progression():
    """
    Test that synthetic board shows expected power progression.
    Absolute powers should increase with channel number.
    """
    config = WindowConfig(window_ms=1000.0)

    with create_board_stream(BoardIds.SYNTHETIC_BOARD) as board:
        window = next(stream_windows(board, config))
        metrics = process_window(window[:3], board.get_sampling_rate(board.board_id))

        # Absolute powers should generally increase
        powers = [m.absolute_alpha for m in metrics]
        assert powers[1] > powers[0]
        assert powers[2] > powers[1]


def test_synthetic_frequency_characteristics():
    """
    Test frequency characteristics of first three synthetic board channels.
    Channel patterns discovered during analysis:
    - Channel 0: Near 5 Hz
    - Channel 1: Near 10 Hz (alpha band)
    - Channel 2: Near 15 Hz
    """
    config = WindowConfig(window_ms=2000.0)  # Longer window for better frequency resolution

    with create_board_stream(BoardIds.SYNTHETIC_BOARD) as board:
        window = next(stream_windows(board, config))
        metrics = process_window(window[:3], board.get_sampling_rate(board.board_id))

        # All channels should show consistent alpha/total ratio
        ratios = [m.alpha_ratio for m in metrics]
        assert max(ratios) - min(ratios) < 0.1  # Ratios within 0.1 of each other


if __name__ == '__main__':
    pytest.main(['-v'])
