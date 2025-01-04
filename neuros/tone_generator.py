import logging
from fluidsynth import Synth

# FluidSynth configuration
SOUNDFONT_PATH = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
MIDI_NOTE = 60  # Middle C
ORGAN_PRESET = 19  # Church Organ

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class ToneGenerator:
    """FluidSynth-based tone generator with continuous organ tones."""

    def __init__(self):
        """Initialize FluidSynth with organ sound"""
        self.fs = Synth()

        # Configure synthesizer
        self.fs.setting("synth.gain", 1.0)
        self.fs.setting("audio.driver", "pulseaudio")
        self.fs.setting("audio.periods", 4)
        self.fs.setting("audio.period-size", 128)

        # Load soundfont and set up organ
        self.sfid = self.fs.sfload(SOUNDFONT_PATH)
        if self.sfid == -1:
            raise RuntimeError(f"Failed to load soundfont: {SOUNDFONT_PATH}")
        self.fs.sfont_select(0, self.sfid)
        self.fs.program_select(0, self.sfid, 0, ORGAN_PRESET)

        # Start audio output
        self.fs.start()

        self.is_playing = False
        self._current_velocity = 0

    def start(self) -> None:
        """Start playing the continuous tone"""
        if not self.is_playing:
            self.fs.noteon(0, MIDI_NOTE, 127)  # Start note with full velocity
            self.is_playing = True

    def stop(self) -> None:
        """Stop the tone"""
        if self.is_playing:
            self.fs.noteoff(0, MIDI_NOTE)
            self.is_playing = False

    def set_amplitude(self, value: float) -> None:
        """Set the amplitude (0.0 to 1.0) of the tone"""
        if not self.is_playing:
            self.start()

        velocity = int(value * 127)
        if velocity != self._current_velocity:
            self.fs.cc(0, 7, velocity)  # Set channel volume
            self._current_velocity = velocity

    def cleanup(self) -> None:
        """Clean up FluidSynth resources"""
        try:
            self.stop()
            self.fs.delete()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
