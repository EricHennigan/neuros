import logging
import numpy as np
import pandas as pd
import threading
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow import DataFilter, DetrendOperations, FilterTypes
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Iterator, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoardReaderParams:
    """
    Configuration parameters for the thread that grabs raw data from the BCI board
    """
    board_id: BoardIds = BoardIds.SYNTHETIC_BOARD
    brainflow_params: BrainFlowInputParams = None
    window: timedelta = timedelta(seconds=0.5) # Window size
    polling: timedelta = timedelta(milliseconds=50) # Time interval between polls

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.brainflow_params:
            object.__setattr__(self, 'brainflow_params', BrainFlowInputParams())
        if self.window <= timedelta(0):
            raise ValueError("Window duration must be positive")
        if self.window <= self.polling:
            raise ValueError("Polling interval must be less than the window duration")
        if self.polling < timedelta(0):
            raise ValueError("Polling duration must be positive")


class BoardReader(BoardShim):

    def __init__(self, params: BoardReaderParams):
        super().__init__(params.board_id, params.brainflow_params)
        self.params = params
        self.stop_event = threading.Event()
        self.data = self._make_data()

    def get_num_channels(self):
       return BoardShim.get_num_rows(self.board_id)

    def get_num_samples(self):
        samples_per_sec = self.get_sampling_rate(self.board_id)
        return int(samples_per_sec * self.params.window.total_seconds()) + 1

    def _make_data(self):
        num_channels = self.get_num_channels()
        num_samples = self.get_num_samples()
        return np.zeros(shape=(num_samples, num_channels))

    def start_reading(self):
        self.thread = threading.Thread(target=self._read_data, args=())
        self.thread.start()

    def stop_reading(self):
        self.stop_event.set()
        self.thread.join()

    def _read_data(self):
        self.prepare_session()
        self.start_stream()

        self.data = self._make_data()
        num_samples = self.get_num_samples()
        while not self.stop_event.is_set():
            time.sleep(self.params.polling.total_seconds())
            data = self.get_board_data()
            data = np.transpose(data)
            self.data = np.append(self.data, data, axis=0)[-num_samples:]

        self.stop_stream()
        self.release_session()
