from fluidsynth import Synth

SOUNDFONT_PATH = "/usr/share/sounds/sf2/FluidR3_GM.sf2"


class ToneGenerator:
    def __init__(self, midi_note=69):
        """Initialize the tone generator with a default MIDI note."""
        self.midi_note = midi_note
        self.synth = Synth()
        self.synth.start(driver="alsa")  # Use ALSA or the default driver
        self.soundfont_id = self.synth.sfload(SOUNDFONT_PATH)
        self.synth.program_select(0, self.soundfont_id, 0, 0)  # Bank 0, Preset 0

    def set_volume(self, intensity):
        """Map intensity (0-1) to MIDI velocity (0-127) and adjust volume."""
        velocity = int(max(0, min(127, intensity * 127)))
        self.synth.cc(0, 7, velocity)  # MIDI Control Change: Volume (CC 7)

    def play(self):
        """Play the note."""
        self.synth.noteon(0, self.midi_note, 100)  # Channel 0, Velocity 100

    def stop(self):
        """Stop the note and clean up resources."""
        self.synth.noteoff(0, self.midi_note)
        self.synth.delete()
