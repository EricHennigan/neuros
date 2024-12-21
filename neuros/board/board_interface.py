# neuros/board/board_interface.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np


@dataclass
class ChannelConfig:
    """Configuration for a single EEG channel"""
    index: int
    label: Optional[str] = None
    enabled: bool = True


class BoardInterface:
    """Interface for EEG board interaction with channel configuration."""

    def __init__(self, board_id: int = BoardIds.SYNTHETIC_BOARD,
                 channels: Optional[List[ChannelConfig]] = None):
        self.board_id = board_id
        self.channels = channels or [ChannelConfig(i) for i in range(8)]  # Default to 8 channels
        self._board = None
        self._eeg_channels = None

    def start(self):
        """Initialize and start the board."""
        if not self._board:
            params = BrainFlowInputParams()
            self._board = BoardShim(self.board_id, params)
            self._eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            self._board.prepare_session()
            self._board.start_stream()

    def stop(self):
        """Stop and release the board."""
        if self._board and self._board.is_prepared():
            self._board.stop_stream()
            self._board.release_session()
            self._board = None
            self._eeg_channels = None

    def get_data(self, num_samples: int) -> Dict[int, np.ndarray]:
        """
        Get current board data for enabled channels.

        Args:
            num_samples: Number of samples to retrieve

        Returns:
            Dictionary mapping channel indices to their data arrays
        """
        if not self._board:
            raise RuntimeError("Board not started")

        data = self._board.get_current_board_data(num_samples)

        # Create a dictionary of enabled channel data
        channel_data = {}
        for channel in self.channels:
            if channel.enabled:
                idx = self._eeg_channels[channel.index]
                channel_data[channel.index] = data[idx]

        return channel_data

    def get_channel_info(self) -> List[Dict]:
        """Get information about configured channels."""
        return [
            {
                'index': ch.index,
                'label': ch.label or f"Channel {ch.index}",
                'enabled': ch.enabled
            }
            for ch in self.channels
        ]

    @property
    def sampling_rate(self) -> int:
        """Get the board's sampling rate."""
        if not self._board:
            raise RuntimeError("Board not initialized")
        return BoardShim.get_sampling_rate(self.board_id)
