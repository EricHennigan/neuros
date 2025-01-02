# test_waveforms.py
import pytest
import numpy as np
from neuros.audio.waveforms import (
    generate_triangle,
    generate_pulse,
    NoiseGenerator,
    HarmonicGenerator
)


def test_triangle_wave():
    """Test triangle wave generation."""
    t = np.linspace(0, 8 * np.pi, 1000)  # More cycles
    wave = generate_triangle(t)

    assert len(wave) == 1000
    assert -1.0 <= wave.min() <= -0.99  # Explicit extrema checks
    assert 0.99 <= wave.max() <= 1.0

    # Test that we hit both extremes
    assert any(np.isclose(wave, 1.0, rtol=1e-2))
    assert any(np.isclose(wave, -1.0, rtol=1e-2))


def test_pulse_wave():
    """Test pulse wave generation with different duty cycles."""
    t = np.linspace(0, 8 * np.pi, 1000)  # More cycles for reliable testing

    # Test default duty cycle (0.5)
    wave = generate_pulse(t)
    assert len(np.unique(wave)) == 2
    assert -1.0 in wave and 1.0 in wave  # Explicit value check
    assert np.all(np.isin(wave, [-1.0, 1.0]))  # Only these values should exist

    # Test custom duty cycle
    wave = generate_pulse(t, duty_cycle=0.25)
    assert -1.0 in wave and 1.0 in wave
    high_samples = np.sum(wave > 0)
    assert np.isclose(high_samples / len(wave), 0.25, rtol=1e-2)

    # Test invalid duty cycle
    with pytest.raises(ValueError):
        generate_pulse(t, duty_cycle=1.5)


def test_noise_generation():
    """Test noise generator properties."""
    generator = NoiseGenerator(seed=42)
    size = 10000

    # Test white noise
    white = generator.generate_white(size)
    assert len(white) == size
    assert np.all(np.abs(white) <= 1.0)
    assert -0.1 < white.mean() < 0.1  # Should be close to zero

    # Test pink noise
    pink = generator.generate_pink(size)
    assert len(pink) == size
    assert np.all(np.isfinite(pink))  # No NaN or inf values

    # Test reproducibility
    gen1 = NoiseGenerator(seed=42)
    gen2 = NoiseGenerator(seed=42)
    assert np.allclose(
        gen1.generate_white(1000),
        gen2.generate_white(1000)
    )


def test_harmonic_generation():
    """Test harmonic synthesis."""
    generator = HarmonicGenerator()
    t = np.linspace(0, 2 * np.pi, 1000)

    # Test basic harmonic series
    harmonics = [1, 2, 3]
    wave = generator.generate_harmonic_series(t, harmonics)
    assert len(wave) == 1000
    assert np.all(np.abs(wave) <= 1.0)

    # Test with custom amplitudes
    amplitudes = [1.0, 0.5, 0.25]
    wave = generator.generate_harmonic_series(t, harmonics, amplitudes)
    assert len(wave) == 1000
    assert np.all(np.abs(wave) <= 1.0)

    # Test mismatched harmonics and amplitudes
    with pytest.raises(ValueError):
        generator.generate_harmonic_series(t, harmonics, [1.0, 0.5])


def test_organ_tone():
    """Test organ tone generation."""
    generator = HarmonicGenerator()
    t = np.linspace(0, 2 * np.pi, 1000)

    wave = generator.generate_organ_tone(t)
    assert len(wave) == 1000
    assert np.all(np.abs(wave) <= 1.0)
    assert np.all(np.isfinite(wave))


if __name__ == '__main__':
    pytest.main(['-v'])
