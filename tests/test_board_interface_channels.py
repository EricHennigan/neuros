# tests/test_board_interface_channels.py

from neuros.board.board_interface import BoardInterface, ChannelConfig
import numpy as np
import pytest


def test_default_channels():
    """Test board initialization with default channels."""
    board = BoardInterface()
    assert len(board.channels) == 8
    assert all(ch.enabled for ch in board.channels)
    assert all(ch.label is None for ch in board.channels)


def test_custom_channels():
    """Test board with custom channel configuration."""
    channels = [
        ChannelConfig(0, label="Fp1"),
        ChannelConfig(1, label="Fp2", enabled=False),
        ChannelConfig(2, label="F3")
    ]
    board = BoardInterface(channels=channels)

    info = board.get_channel_info()
    assert len(info) == 3
    assert info[0]['label'] == "Fp1"
    assert not info[1]['enabled']
    assert info[2]['label'] == "F3"


def test_data_by_channel():
    """Test getting data with channel configuration."""
    channels = [
        ChannelConfig(0, enabled=True),
        ChannelConfig(1, enabled=False),
        ChannelConfig(2, enabled=True)
    ]
    board = BoardInterface(channels=channels)
    board.start()

    data = board.get_data(num_samples=250)

    # Should only get data for enabled channels
    assert set(data.keys()) == {0, 2}
    # Verify we got numpy arrays with at least some data
    assert all(isinstance(arr, np.ndarray) for arr in data.values())
    assert all(arr.size > 0 for arr in data.values())

    board.stop()


def test_sampling_rate():
    """Test sampling rate property."""
    board = BoardInterface()

    # Should raise error if board not started
    with pytest.raises(RuntimeError):
        _ = board.sampling_rate

    board.start()
    assert board.sampling_rate > 0
    board.stop()