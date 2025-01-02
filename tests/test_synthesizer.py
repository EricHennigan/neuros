# test_synthesizer.py
import pytest
import numpy as np
from neuros.audio.synthesizer import WaveformGenerator, WaveformConfig


def test_config_validation():
    """Test WaveformConfig parameter validation."""
    # Valid configurations
    config = WaveformConfig(frequency=440.0, amplitude=0.5)
    assert config.frequency == 440.0
    assert config.amplitude == 0.5

    # Invalid amplitude
    with pytest.raises(ValueError, match="Amplitude must be between"):
        WaveformConfig(amplitude=1.5)

    # Invalid waveform type
    generator = WaveformGenerator(WaveformConfig(waveform_type='invalid'))
    with pytest.raises(ValueError, match="Unsupported waveform type"):
        generator.generate_samples(1000)


def test_basic_waveform_generation():
    """Test basic properties of generated waveforms."""
    sample_rate = 44100
    frequency = 440.0
    generator = WaveformGenerator(
        WaveformConfig(
            frequency=frequency,
            sample_rate=sample_rate,
            amplitude=1.0
        )
    )

    # Generate one cycle
    samples_per_cycle = int(sample_rate / frequency)
    samples = generator.generate_samples(samples_per_cycle)

    # Check basic properties
    assert len(samples) == samples_per_cycle
    assert samples.dtype == np.float32
    assert np.all(np.abs(samples) <= 1.0)


def test_amplitude_modulation():
    """Test thread-safe amplitude control."""
    generator = WaveformGenerator(WaveformConfig(amplitude=1.0))

    # Test amplitude scaling
    generator.set_amplitude(0.5)
    samples = generator.generate_samples(1000)
    assert np.all(np.abs(samples) <= 0.5)

    # Test amplitude limits
    generator.set_amplitude(1.5)  # Should clip to 1.0
    samples = generator.generate_samples(1000)
    assert np.all(np.abs(samples) <= 1.0)


def test_phase_continuity():
    """Test phase continuity between sample generations."""
    generator = WaveformGenerator(
        WaveformConfig(frequency=1.0, sample_rate=1000)
    )
    generator.set_amplitude(1.0)

    # Generate two consecutive buffers
    buffer1 = generator.generate_samples(500)
    buffer2 = generator.generate_samples(500)

    # Combine buffers
    combined = np.concatenate([buffer1, buffer2])

    # Check for discontinuities at the boundary
    boundary_region = combined[495:505]  # 5 samples before and after
    derivatives = np.diff(boundary_region)

    # No sudden jumps at boundary
    assert np.allclose(
        derivatives[4:6],  # Around boundary
        derivatives.mean(),
        rtol=1e-2
    )


def test_waveform_types():
    """Test different waveform types."""
    frequency = 1.0  # 1 Hz for easy testing
    sample_rate = 1000
    config = WaveformConfig(
        frequency=frequency,
        sample_rate=sample_rate,
        amplitude=1.0
    )
    generator = WaveformGenerator(config)
    generator.set_amplitude(1.0)  # Explicitly set amplitude

    waveform_tests = [
        {
            'type': 'sine',
            'validator': lambda samples: (
                    len(samples) == sample_rate and
                    -1.0 <= samples.min() <= 1.0 and
                    -1.0 <= samples.max() <= 1.0
            )
        },
        {
            'type': 'square',
            'validator': lambda samples: (
                    len(np.unique(samples)) == 2 and
                    -1.0 in samples and
                    1.0 in samples
            )
        },
        {
            'type': 'sawtooth',
            'validator': lambda samples: (
                    len(samples) == sample_rate and
                    -1.0 <= samples.min() <= 1.0 and
                    -1.0 <= samples.max() <= 1.0
            )
        }
    ]

    for waveform_test in waveform_tests:
        config.waveform_type = waveform_test['type']
        samples = generator.generate_samples(sample_rate)
        assert waveform_test['validator'](samples), f"{waveform_test['type']} waveform generation failed"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
