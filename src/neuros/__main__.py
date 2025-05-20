import logging
import math
import sys
import threading
import time

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from neuros.neuroboard import BoardReader, BoardReaderParams, Bands
from neuros.tone import Tone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    params = BoardReaderParams(board_id=BoardIds.SYNTHETIC_BOARD)
    board = BoardReader(params)
    board.start_reading()

    def print_eeg_data():
        for i in range(10):
          d = board.get_eeg_power()
          print(d)
          time.sleep(0.5)
        board.stop_reading()
        sys.exit(-1)
    #print_eeg_data()


    channels = board.get_eeg_channels(board.board_id)
    print('shape', board.rdata.shape)
    print(board.rdata[:, channels].shape)
    print('chans', board.get_board_descr(board.board_id)['eeg_channels'])

    # Fill the data buffer
    time.sleep(1)
    d = board.get_eeg_power()
    print(d)

    # TODO: start threads to produce audio
    toneA = Tone(board, band=Bands.THETA, channel='F8', note=Tone.notes[0]).start()
    toneB = Tone(board, band=Bands.THETA, channel='C4', note=Tone.notes[4]).start()
    toneC = Tone(board, band=Bands.ALPHA, channel='Pz', note=Tone.notes[7]).start()

    time.sleep(10)
    toneA.stop()
    toneB.stop()
    toneC.stop()
    board.stop_reading()


if __name__ == "__main__":
    main()
