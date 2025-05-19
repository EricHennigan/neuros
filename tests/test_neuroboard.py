#!/usr/bin/env python

import numpy as np
import pytest
import threading
import time
from brainflow import BoardIds
from neuros.neuroboard import BoardReaderParams, BoardReader, Bands


@pytest.fixture(scope="function")
def board():
  params = BoardReaderParams(board_id=BoardIds.SYNTHETIC_BOARD)
  return BoardReader(params)

def test_data_initialization(board):
    assert board.rdata.shape[0] == board.get_num_samples()
    assert board.rdata.shape[1] == board.get_num_channels()

    num_data = np.count_nonzero(board.rdata[:,0])
    assert num_data == 0


def test_data_single_iteration_read(board):
    board.start_reading()
    time.sleep(board.params.polling.total_seconds())
    board.stop_reading()

    # board data only partially filled the reading window
    # and appends to the end
    num_data = np.count_nonzero(board.rdata[:,0])
    assert num_data > 0
    end_data = np.count_nonzero(board.rdata[-num_data:,0])
    assert end_data > 0

    assert board.rdata.shape[0] == board.get_num_samples()
    assert board.rdata.shape[1] == board.get_num_channels()


def test_data_filling(board):
    intervals = int(board.params.window / board.params.polling) + 1
    board.start_reading()
    time.sleep(intervals * board.params.polling.total_seconds())
    board.stop_reading()

    # data is completely full, did not change shape
    num_data = np.count_nonzero(board.rdata[:,1])
    assert num_data == len(board.rdata[:,1])
    assert board.rdata.shape[0] == board.get_num_samples()
    assert board.rdata.shape[1] == board.get_num_channels()


def test_power(board):
    # full window of data
    intervals = int(board.params.window / board.params.polling) + 1
    board.start_reading()
    time.sleep(intervals * board.params.polling.total_seconds())
    board.stop_reading()

    # second channel of the synthetic board has a 10Hz signal
    channel = board.get_eeg_names()[1]
    power = board.get_eeg_power()[channel][Bands.ALPHA]
    assert 100 <= power
