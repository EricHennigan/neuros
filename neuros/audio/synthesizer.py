import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
import threading

logger = logging.getLogger(__name__)


@dataclass
class WaveformConfig:
    """Configuration for basic waveform generation.

    Defines parameters for generating a continuous waveform.

    Attributes:
        frequency: Frequency of the waveform in Hz.
        amplitude: Base amplitude of the waveform (0.0 to 1.0).
        sample_rate: Sampling rate in Hz.
        waveform_type: Type of waveform ('sine', 'square', 'sawtooth').

    Raises:
        ValueError: If amplitude is not in range [0.0, 1.0].
    """
    frequency: float = 425.0
    amplitude: float = 1.0
    sample_rate: int = 44100
    waveform_type: str = 'sine'

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.amplitude <= 1.0:
            raise ValueError("Amplitude must be between 0.0 and 1.0")


class WaveformGenerator:
    """Generates continuous waveforms with thread-safe amplitude control.

    This class provides basic waveform generation with phase continuity
    between buffer generations. Supports real-time amplitude modulation
    in a thread-safe manner.

    Attributes:
        config: WaveformConfig defining the waveform parameters.
    """

    def __init__(self, config: Optional[WaveformConfig] = None):
        """Initialize the waveform generator.

        Args:
            config: Configuration for waveform generation. If None, uses defaults.
        """
        self.config = config or WaveformConfig()
        self._current_amplitude = 0.0
        self._phase = 0.0
        self._lock = threading.Lock()

    def _generate_sine(self, t: np.ndarray) -> np.ndarray:
        """Generate sine wave samples.

        Args:
            t: Time points array.

        Returns:
            np.ndarray: Sine wave samples.
        """
        return np.sin(t)

    def _generate_square(self, t: np.ndarray) -> np.ndarray:
        """Generate square wave samples.

        Args:
            t: Time points array.

        Returns:
            np.ndarray: Square wave samples.
        """
        return np.where(np.sin(t) >= 0, 1.0, -1.0)

    def _generate_sawtooth(self, t: np.ndarray) -> np.ndarray:
        """Generate sawtooth wave samples.

        Args:
            t: Time points array.

        Returns:
            np.ndarray: Sawtooth wave samples.
        """
        return 2 * (t / (2 * np.pi) - np.floor(0.5 + t / (2 * np.pi)))

    def _generate_waveform(self, t: np.ndarray) -> np.ndarray:
        """Generate waveform based on configured type.

        Args:
            t: Time points array.

        Returns:
            np.ndarray: Waveform samples.

        Raises:
            ValueError: If waveform_type is not recognized.
        """
        generators = {
            'sine': self._generate_sine,
            'square': self._generate_square,
            'sawtooth': self._generate_sawtooth
        }

        if self.config.waveform_type not in generators:
            valid_types = ", ".join(generators.keys())
            raise ValueError(
                f"Unsupported waveform type: {self.config.waveform_type}. "
                f"Valid types: {valid_types}"
            )

        return generators[self.config.waveform_type](t)

    def generate_samples(self, num_samples: int) -> np.ndarray:
        """Generate waveform samples with current amplitude.

        Generates a continuous waveform maintaining phase between calls.
        Thread-safe amplitude modulation is applied to the samples.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            np.ndarray: Generated waveform samples.
        """
        with self._lock:
            current_amplitude = self._current_amplitude

        # Generate time points maintaining phase continuity
        t = 2 * np.pi * self.config.frequency * (
            np.arange(num_samples) / self.config.sample_rate + self._phase
        )

        # Update phase for next call
        self._phase += num_samples / self.config.sample_rate

        # Generate and apply amplitude
        samples = current_amplitude * self._generate_waveform(t)
        return samples.astype(np.float32)

    def set_amplitude(self, value: float) -> None:
        """Thread-safe amplitude adjustment.

        Args:
            value: New amplitude value (0.0 to 1.0).
        """
        with self._lock:
            normalized = np.clip(value, 0, 1)
            self._current_amplitude = normalized * self.config.amplitude
