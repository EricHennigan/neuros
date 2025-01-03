import pytest
from neuros.audio.output import AudioOutput, OutputConfig


def test_output_config():
    """Test OutputConfig initialization and validation"""
    # Test default values
    config = OutputConfig()
    assert config.sample_rate == 44100
    assert config.device is None
    assert config.buffer_size == 256


def test_device_enumeration():
    """Test device listing functionality"""
    devices = AudioOutput.list_devices()
    assert isinstance(devices, list)
    assert len(devices) > 0, "No audio devices found"

    # Verify device info structure
    required_keys = {'index', 'name', 'channels', 'default_samplerate'}
    for device in devices:
        assert all(key in device for key in required_keys)
        assert isinstance(device['channels'], int)
        assert device['channels'] > 0


def test_initialization():
    """Test basic initialization with default settings"""
    output = AudioOutput()
    assert output.config.sample_rate == 44100
    assert output.device_info is not None
    assert output.stream is None  # Stream shouldn't start automatically


def test_custom_configuration():
    """Test initialization with custom configuration"""
    config = OutputConfig(
        sample_rate=48000,
        buffer_size=512
    )
    output = AudioOutput(config)
    assert output.config.sample_rate == 48000
    assert output.config.buffer_size == 512


@pytest.mark.skip(reason="Only run when testing specific device")
def test_invalid_device():
    """Test handling of invalid device specifications"""
    with pytest.raises(RuntimeError):
        config = OutputConfig(device=9999)  # Invalid device index
        AudioOutput(config)


def test_stream_lifecycle():
    """Test audio stream start/stop operations"""
    output = AudioOutput()

    def dummy_callback(outdata, frames, time, status):
        outdata.fill(0.0)

    # Test stream start
    output.start(dummy_callback)
    assert output.stream is not None
    assert output.stream.active

    # Test double start (should not create new stream)
    original_stream = output.stream
    output.start(dummy_callback)
    assert output.stream is original_stream

    # Test stream stop
    output.stop()
    assert output.stream is None

    # Test double stop (should not raise)
    output.stop()


def test_error_handling():
    """Test error handling in various scenarios"""
    output = AudioOutput()

    # Test invalid callback
    with pytest.raises(RuntimeError):
        output.start(None)

    # Test error in callback
    def bad_callback(outdata, frames, time, status):
        raise Exception("Test error")

    output.start(bad_callback)

    # Let the callback execute (which should raise error)
    import time
    time.sleep(0.1)

    # Stream should be stopped due to error
    assert output.stream is None


def test_stream_parameters():
    """Test stream parameter validation"""
    output = AudioOutput()

    def test_callback(outdata, frames, time, status):
        assert outdata.shape[1] == 1  # Mono output
        assert frames == output.config.buffer_size
        outdata.fill(0.0)

    output.start(test_callback)
    # Let a few callbacks happen
    import time
    time.sleep(0.1)
    output.stop()


if __name__ == "__main__":
    pytest.main(["-v", __file__])

