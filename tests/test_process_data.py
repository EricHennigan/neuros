import pytest
import numpy as np
#from neuros.process_data import (
#    compute_power, extract_all_bands,
#    compute_band_ratios, process_window, PowerMetrics
#)


def generate_test_signal(freq: float, sampling_rate: int, duration_sec: float) -> np.ndarray:
    """Generate a pure sine wave for testing"""
    t = np.linspace(0, duration_sec, int(sampling_rate * duration_sec))
    return np.sin(2 * np.pi * freq * t)


def old_test_extract_band_power():
    """Test power extraction for a single frequency band"""
    sampling_rate = 250
    duration = 1.0
    signal = generate_test_signal(10.0, sampling_rate, duration)

    # 10 Hz should show up in alpha band (8-13 Hz)
    alpha = compute_power(signal, sampling_rate, 8, 13)
    beta = compute_power(signal, sampling_rate, 13, 30)

    assert alpha > beta


def old_test_extract_all_bands():
    """Test extraction of all frequency bands"""
    sampling_rate = 250
    duration = 1.0

    # Generate 10 Hz (alpha) signal
    signal = generate_test_signal(10.0, sampling_rate, duration)
    powers = extract_all_bands(signal, sampling_rate)

    # Check we got all expected bands
    expected_bands = {'delta', 'theta', 'alpha', 'beta', 'gamma', 'total'}
    assert set(powers.keys()) == expected_bands

    # Alpha should be strongest (after total)
    secondary_bands = {'delta', 'theta', 'beta', 'gamma'}
    assert all(powers['alpha'] > powers[band] for band in secondary_bands)
    assert powers['total'] >= powers['alpha']


def old_test_compute_band_ratios():
    """Test computation of band ratios"""
    powers = {
        'delta': 1.0,
        'theta': 2.0,
        'alpha': 4.0,
        'beta': 2.0,
        'gamma': 1.0,
        'total': 10.0
    }

    ratios = compute_band_ratios(powers)

    assert np.isclose(ratios['alpha_theta'], 2.0)
    assert np.isclose(ratios['alpha_beta'], 2.0)
    assert np.isclose(ratios['theta_beta'], 1.0)
    assert np.isclose(ratios['alpha_total'], 0.4)


def old_test_process_window_shapes():
    """Test that processing handles different window shapes correctly"""
    sampling_rate = 250
    window_sizes = [125, 250, 500]  # Different numbers of samples
    freq = 10.0  # Alpha frequency

    for size in window_sizes:
        # Create multi-channel test data
        window = np.vstack([
            generate_test_signal(freq, sampling_rate, size / sampling_rate)
            for _ in range(3)
        ])

        metrics = process_window(window, sampling_rate)

        # Basic shape and type checks
        assert len(metrics) == 3
        assert all(isinstance(m, PowerMetrics) for m in metrics)

        # All channels have same signal, should have similar metrics
        for i in range(1, 3):
            assert np.isclose(metrics[i].absolute_alpha,
                              metrics[0].absolute_alpha,
                              rtol=1e-10)
            assert np.isclose(metrics[i].alpha_ratio,
                              metrics[0].alpha_ratio,
                              rtol=1e-10)


def old_test_power_metrics_ranges():
    """Test that power metrics stay in valid ranges"""
    sampling_rate = 250
    duration = 1.0
    signal = generate_test_signal(10.0, sampling_rate, duration)

    window = signal.reshape(1, -1)  # Make 2D for process_window
    metrics = process_window(window, sampling_rate)[0]

    assert metrics.absolute_alpha >= 0
    assert 0 <= metrics.alpha_ratio <= 1
    assert metrics.alpha_beta_ratio >= 0


if __name__ == '__main__':
    pytest.main(['-v'])
