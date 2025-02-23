import numpy as np
from fluidsynth import Synth


class ToneGenerator:
    """FluidSynth-based tone generator with continuous tones and adjustable volume."""

    def __init__(self, notes=None):
        """Initialize FluidSynth with organ sound"""
        self._soundfont_path = "./Aeolus_Soundfont.sf2"
        self._instrument = 0  # Fully Organic Sound

        """Initialize with array of MIDI notes, defaults to 8-note scale"""
        self._notes = notes if notes is not None else [60, 62, 64, 67, 69, 72, 74, 76]  # C major pentatonic (C D E G A C D E)
        self._is_playing = False
        self._current_velocities = [64] * len(self._notes)

        # Initialize synthesizer
        self.synth = Synth()

        # Configure synthesizer
        self.synth.setting("synth.gain", 1.0)
        self.synth.setting("audio.driver", "pulseaudio")

        # Load soundfont and set up organ
        self.sfid = self.synth.sfload(self._soundfont_path)
        if self.sfid == -1:
            raise RuntimeError(f"Failed to load soundfont: {self._soundfont_path}")
        self.synth.sfont_select(0, self.sfid)
        self.synth.program_select(0, self.sfid, 0, self._instrument)

        # Start the synthesizer engine
        self.synth.start()

        # Play the initial note
        self.start()

    def start(self) -> None:
        """Start playing the continuous tone"""

        if self._is_playing:
            return  # Already playing

        for note, velocity in zip(self._notes, self._current_velocities):
            self.synth.noteon(0, note, velocity)
        self._is_playing = True

    def stop(self) -> None:
        """Stop all notes"""
        if not self._is_playing:
            return  # Already stopped

        for note in self._notes:
            self.synth.noteoff(0, note)
        self._is_playing = False

    def set_velocity(self, amplitudes: np.ndarray) -> None:
        """Set amplitudes for all notes"""

        if len(amplitudes) != len(self._notes):
            raise ValueError(f"Expected {len(self._notes)} amplitudes, got {len(amplitudes)}")

        velocities = (amplitudes * 127).astype(int)
        velocities = np.clip(velocities, 0, 127)

        for i, (velocity, note) in enumerate(zip(velocities, self._notes)):
            if velocity != self._current_velocities[i]:
                self.synth.cc(0, 7 + i, velocity)  # Use different CC for each note
                self._current_velocities[i] = velocity



    def cleanup(self) -> None:
        """Clean up FluidSynth resources"""
        try:
            self.stop()
            self.synth.delete()
        except Exception as e:
            print(f"Error during cleanup: {e}")

