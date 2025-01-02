import logging
import pytest
import numpy as np
from neuros.audio.tuner import (
    get_frequency, NOTE_TO_SEMITONES, SCALES
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def test_get_frequency():
    """Test frequency calculation from semitones"""
    # A440 standard reference
    assert np.isclose(get_frequency(440.0, 0), 440.0)  # A4
    assert np.isclose(get_frequency(440.0, 3), 523.25)  # C5
    assert np.isclose(get_frequency(440.0, -12), 220.0)  # A3

    # A425 (project reference)
    assert np.isclose(get_frequency(425.0, 0), 425.0)
    assert np.isclose(get_frequency(425.0, 12), 850.0)

def test_note_mappings():
    """Test note to semitone mappings"""
    # Test all notes map correctly
    assert NOTE_TO_SEMITONES['A'] == 0
    assert NOTE_TO_SEMITONES['C'] == 3
    assert NOTE_TO_SEMITONES['G'] == 10

    # Test enharmonic equivalents
    assert NOTE_TO_SEMITONES['A#'] == NOTE_TO_SEMITONES['Bb']
    assert NOTE_TO_SEMITONES['C#'] == NOTE_TO_SEMITONES['Db']

def test_scale_definitions():
    """Test scale definitions for completeness and validity"""
    for scale_name, notes in SCALES.items():
        # All notes should be valid
        assert all(note in NOTE_TO_SEMITONES for note in notes)

        # Scales should have reasonable lengths
        assert len(notes) >= 5, f"{scale_name} scale too short"
        assert len(notes) <= 12, f"{scale_name} scale too long"

        # Notes should be unique
        assert len(notes) == len(set(notes)), f"Duplicate notes in {scale_name}"


if __name__ == '__main__':
    pytest.main(['-v', __file__])
