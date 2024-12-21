# neuros.board.board_interface.py

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


class BoardInterface:
    """Basic interface for EEG board interaction."""

    def __init__(self, board_id: int = BoardIds.SYNTHETIC_BOARD):
        self.board_id = board_id
        self._board = None

    def start(self):
        """Initialize and start the board."""
        if not self._board:
            params = BrainFlowInputParams()
            self._board = BoardShim(self.board_id, params)
            self._board.prepare_session()
            self._board.start_stream()

    def stop(self):
        """Stop and release the board."""
        if self._board and self._board.is_prepared():
            self._board.stop_stream()
            self._board.release_session()
            self._board = None

    def get_data(self):
        """Get current board data."""
        if not self._board:
            raise RuntimeError("Board not started")
        return self._board.get_current_board_data(num_samples=250)
