import pytest
import numpy as np
#from neuros.main import WindowConfig, stream_windows, board_stream


def old_test_window_config_creation():
    """Test window configuration validation"""
    # Given: Valid window configurations
    valid_configs = [
        (500.0, 250.0, "Standard window with 50% overlap"),
        (100.0, 60.0, "Small window with partial overlap"),
        (1000.0, 0.0, "No overlap window"),
    ]

    # When/Then: Creating valid configurations succeeds
    for window_ms, overlap_ms, scenario in valid_configs:
        config = WindowConfig(window_ms=window_ms, overlap_ms=overlap_ms)
        assert config.window_ms == window_ms, scenario
        assert config.overlap_ms == overlap_ms, scenario

    # Given: Invalid window configurations with expected error messages
    invalid_configs = [
        (100.0, 100.0, "Overlap must be less than window size"),
        (100.0, 150.0, "Overlap must be less than window size"),
        (0.0, 0.0, "Window size must be positive"),
        (-100.0, 50.0, "Window size must be positive"),
    ]

    # When/Then: Creating invalid configurations raises ValueError with expected message
    for window_ms, overlap_ms, expected_msg in invalid_configs:
        with pytest.raises(ValueError, match=expected_msg):
            WindowConfig(window_ms=window_ms, overlap_ms=overlap_ms)


def old_test_sample_conversion():
    """Test conversion from milliseconds to samples"""
    # Given: A window configuration
    config = WindowConfig(window_ms=500.0, overlap_ms=250.0)
    sample_rate = 250  # Hz

    # When: Converting to samples
    window_samples, overlap_samples = config.to_samples(sample_rate)

    # Then: Sample counts should match expected values
    assert window_samples == 125, "500ms at 250Hz should be 125 samples"
    assert overlap_samples == 62, "250ms at 250Hz should be 62 samples"


def old_test_window_generation():
    """Test basic window generation functionality"""
    # Given: A standard window configuration
    config = WindowConfig(window_ms=500.0, overlap_ms=250.0)

    with board_stream() as board:
        # When: Generating windows
        generator = stream_windows(board, config)
        window = next(generator)

        # Then: Window shape should match configuration
        sample_rate = board.get_sampling_rate(board_id=board.board_id)
        expected_samples = int(sample_rate * config.window_ms / 1000)
        eeg_channels = board.get_eeg_channels(board_id=board.board_id)

        assert isinstance(window, np.ndarray), "Window should be numpy array"
        assert window.shape == (len(eeg_channels), expected_samples), "Window shape incorrect"
        assert not np.any(np.isnan(window)), "Window should not contain NaN values"


def old_test_window_overlap():
    """Test window overlap behavior"""
    # Given: A configuration with 50% overlap
    config = WindowConfig(window_ms=500.0, overlap_ms=250.0)

    with board_stream() as board:
        generator = stream_windows(board, config)

        # When: Generating consecutive windows
        window1 = next(generator)
        window2 = next(generator)

        # Then: Windows should overlap by the specified amount
        sample_rate = board.get_sampling_rate(board_id=board.board_id)
        overlap_samples = int(sample_rate * config.overlap_ms / 1000)

        np.testing.assert_array_almost_equal(
            window1[:, -overlap_samples:],
            window2[:, :overlap_samples],
            decimal=5,
            err_msg="Overlap regions should match"
        )


def old_test_data_continuity():
    """Test that window data progresses over time"""
    # Given: A standard configuration
    config = WindowConfig(window_ms=500.0, overlap_ms=250.0)

    with board_stream() as board:
        generator = stream_windows(board, config)

        # When: Collecting multiple windows
        windows = []
        for _ in range(3):
            windows.append(next(generator))

        # Then: Windows should contain different but valid data
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                assert not np.array_equal(windows[i], windows[j]), \
                    "Consecutive windows should contain different data"
                assert np.all(np.isfinite(windows[i])), \
                    "All values should be finite"
                assert np.all(np.abs(windows[i]) < 1000), \
                    "Values should be within reasonable range"


def old_test_window_params_no_common_factors():
    """Test window generation with parameters that round to unfriendly numbers"""
    # Given: A configuration with parameters that round down to numbers with no common factors
    # 431ms at 250Hz = 107.75 samples -> 107 samples
    # 167ms at 250Hz = 41.75 samples -> 41 samples
    config = WindowConfig(window_ms=431.0, overlap_ms=167.0)

    with board_stream() as board:
        # When: Generating windows
        generator = stream_windows(board, config)
        window = next(generator)

        # Then: Window shape should match rounded sample counts
        sample_rate = board.get_sampling_rate(board_id=board.board_id)
        expected_samples = int(sample_rate * config.window_ms / 1000)
        assert expected_samples == 107, "Window size should be rounded down"
        eeg_channels = board.get_eeg_channels(board_id=board.board_id)

        assert isinstance(window, np.ndarray), "Window should be numpy array"
        assert window.shape == (len(eeg_channels), expected_samples), \
            f"Window shape incorrect. Expected {(len(eeg_channels), expected_samples)}, got {window.shape}"
        assert not np.any(np.isnan(window)), "Window should not contain NaN values"

        # Check for data repetition (synthetic board artifact)
        window2 = next(generator)
        assert not np.array_equal(window, window2), "Windows should contain different data"

        # Verify overlap
        overlap_samples = int(sample_rate * config.overlap_ms / 1000)
        assert overlap_samples == 41, "Overlap size should be rounded down"
        np.testing.assert_array_almost_equal(
            window[:, -overlap_samples:],
            window2[:, :overlap_samples],
            decimal=5,
            err_msg="Overlap regions should match with prime-like sample counts"
        )


def old_test_error_handling():
    """Test recovery from board interruption"""
    # Given: A running board stream
    config = WindowConfig(window_ms=500.0, overlap_ms=250.0)

    with board_stream() as board:
        generator = stream_windows(board, config)

        # Initially verify normal operation
        window1 = next(generator)
        initial_shape = window1.shape
        assert np.all(np.isfinite(window1)), "Initial data should be valid"

        # When: Board stream is interrupted
        board.stop_stream()

        # Then: Generator should handle interruption gracefully
        window2 = next(generator)
        assert isinstance(window2, np.ndarray), "Should return valid array type"
        assert window2.shape == initial_shape, "Should maintain consistent shape"
        assert np.all(np.isfinite(window2)), "Should contain valid data"

        # When: Stream is restored
        board.start_stream()

        # Then: Generator should resume normal operation
        window3 = next(generator)
        assert isinstance(window3, np.ndarray), "Should return valid array type"
        assert window3.shape == initial_shape, "Should maintain consistent shape"
        assert np.all(np.isfinite(window3)), "Should contain valid data"
        assert not np.array_equal(window3, window2), "Should contain new data"


def old_test_startup_behavior():
    """Test behavior during initial data accumulation"""
    # Given: A window configuration requiring multiple chunks
    config = WindowConfig(window_ms=1000.0, overlap_ms=0.0)

    with board_stream() as board:
        generator = stream_windows(board, config)

        # When: Requesting first window
        window = next(generator)

        # Then: Should receive complete, valid window
        sample_rate = board.get_sampling_rate(board_id=board.board_id)
        expected_samples = int(sample_rate * config.window_ms / 1000)
        assert window.shape[1] == expected_samples, "Should wait for complete window"
        assert np.all(np.isfinite(window)), "Should contain valid data"


def old_test_resource_cleanup():
    """Test proper resource management"""
    # Given: A stream configuration
    config = WindowConfig(window_ms=500.0, overlap_ms=250.0)

    # When: Using and exiting the board context
    with board_stream() as board:
        generator = stream_windows(board, config)
        _ = next(generator)
        assert board.is_prepared(), "Board should be prepared during use"

    # Then: Resources should be properly cleaned up
    assert not board.is_prepared(), "Board should be cleaned up after use"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
