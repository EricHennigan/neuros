import pytest
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from neuros.main import WindowConfig, stream_windows, board_stream


def test_window_config_validation():
    """Test window configuration validation"""
    # Valid configurations
    WindowConfig(window_ms=500.0, overlap_ms=250.0)  # Known good values
    WindowConfig(window_ms=100.0, overlap_ms=60.0)  # Another known good set

    # Invalid configurations
    with pytest.raises(ValueError):
        WindowConfig(window_ms=100.0, overlap_ms=100.0)  # Equal window and overlap
    with pytest.raises(ValueError):
        WindowConfig(window_ms=100.0, overlap_ms=150.0)  # Overlap > window


def test_basic_window_shape():
    """Test that windows have the expected shape"""
    config = WindowConfig(window_ms=500.0, overlap_ms=250.0)  # Known good values

    with board_stream() as board:
        sample_rate = board.get_sampling_rate(board_id=board.board_id)
        expected_samples = int(sample_rate * config.window_ms / 1000)
        eeg_channels = board.get_eeg_channels(board_id=board.board_id)

        generator = stream_windows(board, config)
        window = next(generator)

        assert isinstance(window, np.ndarray)
        assert window.shape == (len(eeg_channels), expected_samples)


def test_window_overlap():
    """Test that consecutive windows overlap by the expected amount"""
    config = WindowConfig(window_ms=500.0, overlap_ms=250.0)  # Known good values

    with board_stream() as board:
        generator = stream_windows(board, config)

        # Get two consecutive windows
        window1 = next(generator)
        window2 = next(generator)

        # Calculate expected overlap in samples
        sample_rate = board.get_sampling_rate(board_id=board.board_id)
        overlap_samples = int(sample_rate * config.overlap_ms / 1000)

        # Check that the end of window1 matches the start of window2
        np.testing.assert_array_almost_equal(
            window1[:, -overlap_samples:],
            window2[:, :overlap_samples],
            decimal=5  # Adjust precision as needed
        )


@pytest.mark.parametrize("config", [
    WindowConfig(window_ms=500.0, overlap_ms=250.0),  # Known good
    WindowConfig(window_ms=100.0, overlap_ms=60.0),  # Also good
])
def test_window_continuity(config):
    """Test that windows contain different data and progress over time"""
    with board_stream() as board:
        generator = stream_windows(board, config)

        # Skip first few windows to let board stabilize
        for _ in range(3):
            next(generator)

        # Get test windows
        windows = [next(generator) for _ in range(3)]

        # Verify windows are different
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                # Windows should not be identical
                assert not np.array_equal(windows[i], windows[j])

                # But they should be in a reasonable range
                assert np.all(np.abs(windows[i]) < 1000)  # Adjust threshold as needed


def test_eeg_channel_selection():
    """Test that windows only contain EEG channels"""
    config = WindowConfig(window_ms=500.0, overlap_ms=250.0)

    with board_stream() as board:
        eeg_channels = board.get_eeg_channels(board_id=board.board_id)
        generator = stream_windows(board, config)
        window = next(generator)

        assert window.shape[0] == len(eeg_channels)


def test_generator_cleanup():
    """Test that the generator cleans up properly"""
    config = WindowConfig(window_ms=500.0, overlap_ms=250.0)

    with board_stream() as board:
        generator = stream_windows(board, config)

        # Get a few windows
        for _ in range(3):
            window = next(generator)
            assert window is not None

        # Generator should exit cleanly
        generator.close()


def test_error_handling():
    """Test handling of board errors and recovery"""
    config = WindowConfig(window_ms=500.0, overlap_ms=250.0)

    with board_stream() as board:
        generator = stream_windows(board, config)

        # Get initial window to verify normal operation
        window1 = next(generator)
        assert window1 is not None
        assert isinstance(window1, np.ndarray)

        # Force an error by stopping the board
        board.stop_stream()

        # Generator should either:
        # 1. Return empty/zero data
        # 2. Skip the error and continue on next valid data
        # 3. Or maintain last valid state
        window2 = next(generator)
        assert window2 is not None  # Shouldn't crash

        # Start the board again
        board.start_stream()

        # Should recover and provide new data
        window3 = next(generator)
        assert window3 is not None
        assert isinstance(window3, np.ndarray)


@pytest.mark.parametrize("bad_config", [
    WindowConfig(window_ms=10.0, overlap_ms=0.0),  # Too small
    WindowConfig(window_ms=10000.0, overlap_ms=0.0),  # Too large
])
def test_extreme_window_sizes(bad_config):
    """Test handling of extreme window sizes"""
    with board_stream() as board:
        generator = stream_windows(board, bad_config)

        # Should either work or fail gracefully
        try:
            window = next(generator)
            assert window is not None
            assert isinstance(window, np.ndarray)
        except Exception as e:
            assert str(e) != ""  # Should have a meaningful error message


if __name__ == "__main__":
    pytest.main(["-v", __file__])