from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Note mapping using a type-safe approach
NoteMapping = Dict[str, int]
ScaleMapping = Dict[str, List[str]]

# Map note names to semitones from A
NOTE_TO_SEMITONES: NoteMapping = {
    'A': 0,
    'A#': 1, 'Bb': 1,
    'B': 2,
    'C': 3,
    'C#': 4, 'Db': 4,
    'D': 5,
    'D#': 6, 'Eb': 6,
    'E': 7,
    'F': 8,
    'F#': 9, 'Gb': 9,
    'G': 10,
    'G#': 11, 'Ab': 11
}

# Standard musical scales
SCALES: ScaleMapping = {
    'pentatonic': ['A', 'C', 'D', 'E', 'G'],
    'major': ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'],
    'minor': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    'chromatic': ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
}


def get_frequency(base_a_freq: float, semitones: int) -> float:
    """Calculate frequency relative to base A using equal temperament.

    Args:
        base_a_freq: Base frequency for A note in Hz (typically 425.0)
        semitones: Number of semitones away from A (positive or negative)

    Returns:
        float: Calculated frequency in Hz

    Example:
        >>> get_frequency(425.0, 3)  # Calculate C frequency
        505.53...
    """
    logger.debug(f"Calculating frequency: base={base_a_freq}Hz, semitones={semitones}")
    return base_a_freq * (2 ** (semitones / 12))


def get_scale_frequencies(
        scale_name: str,
        base_a_freq: float = 425.0,
        octave_shift: int = -1
) -> List[float]:
    """Generate frequencies for a musical scale.

    Args:
        scale_name: Name of the scale ('pentatonic', 'major', 'minor', 'chromatic')
        base_a_freq: Base frequency for A note in Hz (default: 425.0)
        octave_shift: Number of octaves to shift (default: -1)

    Returns:
        List[float]: List of frequencies in Hz for each note in the scale

    Raises:
        ValueError: If scale_name is not recognized

    Example:
        >>> freqs = get_scale_frequencies('pentatonic', 425.0, -1)
        >>> len(freqs)
        5
    """
    if scale_name not in SCALES:
        valid_scales = ", ".join(SCALES.keys())
        raise ValueError(f"Unknown scale: {scale_name}. Valid scales: {valid_scales}")

    # Apply octave shift to base frequency
    actual_base_freq = base_a_freq * (2 ** octave_shift)
    logger.debug(f"Using scale '{scale_name}' with base freq {actual_base_freq}Hz")

    # Convert note names to frequencies
    frequencies = [
        get_frequency(actual_base_freq, NOTE_TO_SEMITONES[note])
        for note in SCALES[scale_name]
    ]

    return frequencies
