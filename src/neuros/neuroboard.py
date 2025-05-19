import logging
import numpy as np
import pandas as pd
import threading
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow import DataFilter, DetrendOperations, FilterTypes, WindowOperations
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Iterator, Optional, Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoardReaderParams:
    """
    Configuration parameters for the thread that grabs raw data from the BCI board
    """
    board_id: BoardIds = BoardIds.SYNTHETIC_BOARD
    brainflow_params: BrainFlowInputParams = None
    window: timedelta = timedelta(seconds=1) # Window size
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
        self.stop_event = threading.Event() # external signal to stop reading data from board
        self._update = threading.Event() # internal signal new data has been read
        self.rdata = self._make_data() # raw data from the board
        self.pdata = self._make_data() # processed data (psd)

    def get_num_channels(self) -> int:
        return BoardShim.get_num_rows(self.board_id)

    def get_sampling_rate(self) -> float:
        '''sampling rate in samples per second'''
        return BoardShim.get_sampling_rate(self.board_id)

    def get_num_samples(self) -> int:
        samples_per_sec = self.get_sampling_rate()
        return int(samples_per_sec * self.params.window.total_seconds()) + 1

    def _make_data(self) -> np.ndarray:
        '''2D array shape=(num_samples, num_channels)'''
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

        num_samples = self.get_num_samples()
        while not self.stop_event.is_set():
            time.sleep(self.params.polling.total_seconds())
            rdata = self.get_board_data()
            rdata = np.transpose(rdata)
            self.rdata = np.append(self.rdata, rdata, axis=0)[-num_samples:]
            self._update.set()

        self.stop_stream()
        self.release_session()

    def get_eeg_names(self) -> List[str]:
        return BoardShim.get_eeg_names(self.board_id)

    # NOTE: possible optimization to transpose the data?
    # CAUTION: this method takes about 20ms to execute
    def get_eeg_power(self) -> pd.DataFrame:
        '''Returns brainwave band power of the EEG channels'''
        if not self._update.is_set():
            return self.pdata

        sample_rate = self.get_sampling_rate()
        channels = self.get_eeg_channels(self.board_id)
        names = self.get_eeg_names()

        # work on copy of the raw data, trimmed to the eeg channels
        rdata = self.rdata[:, channels]
        size = self.get_num_samples()
        size = size-1 if size%2 else size # truncate to nearest even number

        # compute the power spectral density
        pdata = pd.DataFrame(columns=names, index=[b for b in Bands])
        for c in range(len(channels)):
            psd = DataFilter.get_psd(rdata[:, c][:size], sample_rate, WindowOperations.BLACKMAN_HARRIS)
            for b in Bands:
                pdata.loc[b, names[c]] = DataFilter.get_band_power(psd, *b.value)

        self.pdata = pdata
        self._update.clear()
        return self.pdata


class Bands(Enum):
    DELTA = (0.5, 4.0)
    THETA = (4.0, 8.0)
    ALPHA = (8.0, 13.0)
    BETA = (13.0, 30.0)
    GAMMA = (30.0, 50.0) # below 60Hz to avoid ambient electricity
    ALL = (0.5, 50.0)

