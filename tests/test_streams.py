import pytest
import numpy as np
import time
from brainflow.board_shim import BoardIds
from neuros.main import WindowConfig, board_stream, stream_windows


def test_window_config():
    """Test window configuration validation and conversion"""
    # Valid configs
    config = WindowConfig(window_ms=100.0, overlap_ms=50.0)
    window_samples, overlap_samples = config.to_samples(sample_rate=250)
    assert window_samples == 25  # 100ms * 250Hz / 1000
    assert overlap_samples == 12  # 50ms * 250Hz / 1000

    # Invalid overlap
    with pytest.raises(ValueError):
        config = WindowConfig(window_ms=100.0, overlap_ms=100.0)
        config.to_samples(250)


def test_board_context():
    """Test board context manager handles resources properly"""
    # Normal exit
    with board_stream() as board:
        assert board.is_prepared()
        # Brief pause to let board start up
        time.sleep(0.1)
        data = board.get_current_board_data(100)
        assert data.size > 0
    assert not board.is_prepared()

    # Exception handling
    with pytest.raises(ValueError):
        with board_stream() as board:
            raise ValueError("Test exception")
    assert not board.is_prepared()


def test_stream_basic():
    """Test basic window streaming functionality"""
    config = WindowConfig(window_ms=200.0)  # 200ms windows, no overlap

    with board_stream() as board:
        # Let board generate some data
        time.sleep(0.5)

        # Get first few windows
        windows = []
        for window in stream_windows(board, config):
            windows.append(window)
            if len(windows) >= 3:
                break

        # Check window properties
        sample_rate = board.get_sampling_rate(board_id=BoardIds.SYNTHETIC_BOARD)
        expected_samples = int(0.2 * sample_rate)  # 200ms worth

        assert len(windows) == 3
        for window in windows:
            assert isinstance(window, np.ndarray)
            assert window.ndim == 2  # (channels, samples)
            assert window.shape[1] == expected_samples


def test_stream_overlap():
    """Test overlapping windows work correctly"""
    config = WindowConfig(window_ms=100.0, overlap_ms=50.0)

    with board_stream() as board:
        # Let board generate some data
        time.sleep(0.5)

        # Collect a few windows
        windows = []
        for window in stream_windows(board, config):
            windows.append(window)
            if len(windows) >= 4:
                break

        # With 50% overlap, second half of each window should match
        # first half of next window
        for w1, w2 in zip(windows[:-1], windows[1:]):
            half_size = w1.shape[1] // 2
            np.testing.assert_array_almost_equal(
                w1[:, -half_size:],
                w2[:, :half_size]
            )


def test_stream_empty_board():
    """Test handling of empty data from board"""
    config = WindowConfig(window_ms=100.0)

    with board_stream() as board:
        # Try to get windows immediately (board likely has no data yet)
        windows = []
        start_time = time.time()
        timeout = 1.0  # 1 second timeout

        for window in stream_windows(board, config):
            windows.append(window)
            if len(windows) >= 2 or (time.time() - start_time) > timeout:
                break

        # Should eventually get some windows
        assert len(windows) > 0


def test_eeg_channel_count():
    """Test that we get expected number of channels"""
    config = WindowConfig(window_ms=100.0)

    with board_stream(BoardIds.SYNTHETIC_BOARD) as board:
        time.sleep(0.5)  # Let board generate data

        # Get one window
        window = next(stream_windows(board, config))

        # We get all channels from the board, not just EEG
        # Synthetic board has 16 EEG channels and 16 other channels
        assert window.shape[0] == 16  # Total number of eeg channels for synthetic board


def test_ring_buffer_shift():
    # Create a known pattern in the ring buffer
    ring_buffer = np.array([[1, 2, 3, 4, 5, 6]])  # Single channel for simplicity
    new_data = np.array([[7, 8, 9]])
    new_samples = 3

    # Perform the shift
    ring_buffer[:, :-new_samples] = ring_buffer[:, new_samples:]
    ring_buffer[:, -new_samples:] = new_data

    # Check if data shifted correctly
    expected = np.array([[4, 5, 6, 7, 8, 9]])
    np.testing.assert_array_equal(ring_buffer, expected)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
