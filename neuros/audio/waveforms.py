import logging
from typing import Optional, List
import numpy as np

logger = logging.getLogger(__name__)


def generate_triangle(t: np.ndarray) -> np.ndarray:
    """Generate triangle wave samples.

    Uses a frequency-independent method that maintains waveform symmetry.

    Args:
        t: Time points array (phase in radians).

    Returns:
        np.ndarray: Triangle wave samples in range [-1, 1].
    """
    return 2 * np.abs(2 * (t / (2 * np.pi) - np.floor(0.5 + t / (2 * np.pi)))) - 1


def generate_pulse(t: np.ndarray, duty_cycle: float = 0.5) -> np.ndarray:
    """Generate pulse wave samples.

    Args:
        t: Time points array (phase in radians).
        duty_cycle: Portion of cycle spent high (0.0 to 1.0).

    Returns:
        np.ndarray: Pulse wave samples in range [-1, 1].

    Raises:
        ValueError: If duty_cycle is not between 0 and 1.
    """
    if not 0.0 <= duty_cycle <= 1.0:
        raise ValueError("Duty cycle must be between 0.0 and 1.0")
    return np.where(np.mod(t / (2 * np.pi), 1.0) < duty_cycle, 1.0, -1.0)


class NoiseGenerator:
    """Generates various types of noise with consistent state."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize noise generator.

        Args:
            seed: Random seed for reproducible noise.
        """
        self._rng = np.random.RandomState(seed)

    def generate_white(self, size: int) -> np.ndarray:
        """Generate white noise samples.

        Args:
            size: Number of samples to generate.

        Returns:
            np.ndarray: White noise samples in range [-1, 1].
        """
        return self._rng.uniform(-1.0, 1.0, size=size)

    def generate_pink(self, size: int) -> np.ndarray:
        """Generate pink noise samples.

        Uses Voss-McCartney algorithm for efficient generation.

        Args:
            size: Number of samples to generate.

        Returns:
            np.ndarray: Pink noise samples, normalized to range [-1, 1].
        """
        # Voss-McCartney algorithm implementation
        white = self.generate_white(size)
        pink = np.zeros_like(white)

        # Use 8 octaves for approximation
        octaves = 8
        for i in range(octaves):
            # Calculate number of values needed at this octave
            octave_size = size // (2 ** i)
            if octave_size < 1:
                break

            # Generate and repeat values
            values = self._rng.randn(octave_size)
            repeated = np.repeat(values, 2 ** i)
            # Trim or pad to match size
            if len(repeated) > size:
                repeated = repeated[:size]
            elif len(repeated) < size:
                repeated = np.pad(repeated, (0, size - len(repeated)), 'edge')

            pink += repeated

        # Normalize to [-1, 1]
        pink /= np.max(np.abs(pink))

        return pink


class HarmonicGenerator:
    """Generates complex waveforms using harmonic synthesis."""

    def generate_harmonic_series(
            self,
            t: np.ndarray,
            harmonics: List[float],
            amplitudes: Optional[List[float]] = None
    ) -> np.ndarray:
        """Generate waveform from harmonic series.

        Args:
            t: Time points array (phase in radians).
            harmonics: List of harmonic multipliers.
            amplitudes: List of amplitudes for each harmonic.
                If None, uses 1/n falloff.

        Returns:
            np.ndarray: Combined harmonic waveform, normalized to range [-1, 1].

        Raises:
            ValueError: If harmonics and amplitudes lengths don't match.
        """
        if amplitudes is None:
            amplitudes = [1.0 / (i + 1) for i in range(len(harmonics))]

        if len(harmonics) != len(amplitudes):
            raise ValueError("Number of harmonics must match number of amplitudes")

        # Generate and sum harmonics
        wave = np.zeros_like(t)
        for harmonic, amplitude in zip(harmonics, amplitudes):
            wave += amplitude * np.sin(harmonic * t)

        # Normalize
        return wave / np.max(np.abs(wave))

    def generate_organ_tone(self, t: np.ndarray) -> np.ndarray:
        """Generate a basic organ-like tone.

        Uses a specific combination of harmonics typical of organ pipes.

        Args:
            t: Time points array (phase in radians).

        Returns:
            np.ndarray: Organ-like waveform samples.
        """
        # Typical organ pipe harmonics
        harmonics = [1, 2, 3, 4, 6, 8]  # Fundamental and overtones
        amplitudes = [1.0, 0.6, 0.4, 0.3, 0.2, 0.1]  # Characteristic falloff

        return self.generate_harmonic_series(t, harmonics, amplitudes)

# Possible Future additions:
# - FM synthesis
# - Additive synthesis with phase control
# - More noise colors (brown, violet)
# - Waveshaping
