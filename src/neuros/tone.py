import math
import threading
import time

from fluidsynth import Synth
from importlib.resources import files
from neuros.neuroboard import BoardReader, Bands

# Fluidsynth constants
VOLUME = 7

class Tone(object):
    notes = [60, 62, 64, 67, 69, 72, 74, 76] # C major pentatonic (C D E G A C D E)

    synth = Synth()
    synth.start(driver = 'pulseaudio')
    synth_fid = synth.sfload(str(files('neuros.data').joinpath('Aeolus_Soundfont.sf2')))
    synth.setting('synth.gain', 1.0)

    cid = 17 # midi channel ids

    def __init__(self, board: BoardReader, band: Bands, channel: str|int, note: int):
        # Note: Channel is an EEG Channel name
        self.board = board
        self.band = band
        if type(channel) == int:
          channel = board.get_eeg_names()[channel]
        self.channel = channel
        self.note = note
        self.thread = threading.Thread(target=self._play_note, args=())
        self._playing = threading.Event()
        self.cid = Tone.cid
        Tone.cid += 1

    def start(self):
        Tone.synth.program_select(self.cid, Tone.synth_fid, 0, 0)
        Tone.synth.noteon(self.cid, self.note, 64)
        self._playing.set()
        self.thread.start()
        return self

    def stop(self):
        Tone.synth.noteoff(self.cid, self.note)
        self._playing.clear()
        self.thread.join()

    def _play_note(self):
        # Play a note in the upper half of the volume range
        vmax = 0.0
        while self._playing.is_set():
            v = self.board.get_eeg_power()[self.channel][self.band]
            vmax = max(v, vmax)
            vol = 64 + 64 * (v / vmax)
            Tone.synth.cc(self.cid, VOLUME, int(vol))
            time.sleep(0.005)

