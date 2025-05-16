#!/usr/bin/env python

import threading
import time
import pytest
import numpy as np
from brainflow import BoardIds, BoardShim, BrainFlowInputParams
from neuros.eeg_reader import raw_data, WindowParams


@pytest.fixture(scope="function")
def board():
  board_id = BoardIds.SYNTHETIC_BOARD
  board_params = BrainFlowInputParams()
  board = BoardShim(board_id, board_params)

  data_params = WindowParams()
  board.data_thread = threading.Thread(target=raw_data, args=(board, data_params))
  board._data_params = data_params
  return board

def start(board):
  board.data_thread.start()

def test_window_size(board):
  print('stop1?', board._data_params.stop_event)
  board._data_params.stop_event.set()
  start(board)
  board.data_thread.join()

  samples_per_sec = board.get_sampling_rate(board.board_id)
  window_sec = board._data_params.window.total_seconds()
  num_samples = int(samples_per_sec * window_sec) + 1
  assert board._data.shape[0] == num_samples

  num_data = np.count_nonzero(board._data[:,0])
  assert num_data == 0


def test_data_initialization(board):
  start(board)
  time.sleep(board._data_params.polling.total_seconds())
  board._data_params.stop_event.set()
  board.data_thread.join()

  num_data = np.count_nonzero(board._data[:,0])
  assert num_data > 0
  end_data = np.count_nonzero(board._data[-num_data:,0])
  assert end_data > 0


def test_data_filling(board):
  intervals = int(board._data_params.window / board._data_params.polling) + 1
  start(board)
  time.sleep(intervals * board._data_params.polling.total_seconds())
  board._data_params.stop_event.set()
  board.data_thread.join()

  num_data = np.count_nonzero(board._data[:,0])
  assert num_data == len(board._data[:,0])
