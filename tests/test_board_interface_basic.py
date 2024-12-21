# tests/test_board_interface_basic.py

from neuros.board.board_interface import BoardInterface
import pytest
import time


def test_board_lifecycle():
    """Test basic board start/stop functionality."""
    board = BoardInterface()

    # Should start without error
    board.start()
    assert board._board is not None

    # Should be able to get data
    data = board.get_data()
    assert data is not None
    assert data.shape[1] > 0  # Should have some samples

    # Should stop cleanly
    board.stop()
    assert board._board is None


def test_data_acquisition():
    """Test that we can actually get changing data."""
    board = BoardInterface()
    board.start()

    # Get two samples with a small delay
    data1 = board.get_data()
    time.sleep(0.1)  # Small delay
    data2 = board.get_data()

    # Data should be different
    assert not (data1 == data2).all()

    board.stop()


def test_error_handling():
    """Test error conditions."""
    board = BoardInterface()

    # Should raise error if trying to get data before starting
    with pytest.raises(RuntimeError):
        board.get_data()

    # Should handle multiple stops gracefully
    board.start()
    board.stop()
    board.stop()  # Should not raise error