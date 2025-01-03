import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
import sounddevice as sd

logger = logging.getLogger(__name__)


@dataclass
class OutputConfig:
    """Configuration for audio output.

    Attributes:
        sample_rate: Sampling rate in Hz
        device: Optional device index (None for system default)
        buffer_size: Audio buffer size in frames
    """
    sample_rate: int = 44100
    device: Optional[int] = None
    buffer_size: int = 256  # Reduced for lower latency


class AudioOutput:
    """Manages audio device setup and output streaming.

    This class handles:
    - Audio device enumeration and validation
    - Device setup and fallback logic
    - Stream lifecycle management
    """

    def __init__(self, config: Optional[OutputConfig] = None):
        """Initialize audio output system.

        Args:
            config: Output configuration. Uses defaults if None.

        Raises:
            RuntimeError: If no valid output device can be found.
        """
        self.config = config or OutputConfig()
        self.stream: Optional[sd.OutputStream] = None
        self.device_info = None
        self._setup_device()

    def _setup_device(self) -> None:
        """Validate and setup the audio device.

        Raises:
            RuntimeError: If device initialization fails.
        """
        try:
            # Try specified device or get system default
            if self.config.device is not None:
                self.device_info = sd.query_devices(self.config.device)
            else:
                self.device_info = sd.query_devices(kind='output')
                self.config.device = self.device_info['index']

            # Validate device capabilities
            if self.device_info['max_output_channels'] < 1:
                raise RuntimeError("Device has no output channels")

            logger.info(
                f"Using audio device: {self.device_info['name']} "
                f"({self.device_info['max_output_channels']} channels)"
            )

        except Exception as e:
            logger.error(f"Device setup failed: {e}")
            raise RuntimeError(f"Could not initialize audio device: {e}")

    @staticmethod
    def list_devices() -> List[Dict]:
        """List all available audio output devices.

        Returns:
            List of device information dictionaries.
        """
        devices = sd.query_devices()
        print(devices)
        return [
            {
                'index': dev['index'],
                'name': dev['name'],
                'channels': dev['max_output_channels'],
                'default_samplerate': dev['default_samplerate']
            }
            for dev in devices
            if dev['max_output_channels'] > 0
        ]

    def start(self, callback: Callable) -> None:
        """Start the audio stream.

        Args:
            callback: Audio processing callback function.
                     Must accept (outdata, frames, time, status) parameters.

        Raises:
            RuntimeError: If callback is invalid or stream fails to start.
        """
        if self.stream is not None:
            return

        if not callable(callback):
            raise RuntimeError("Invalid callback - must be callable")

        def wrapped_callback(outdata, frames, time, status):
            """Wrap the callback to handle errors and stop the stream if needed."""
            try:
                callback(outdata, frames, time, status)
                if status and status.output_underflow:
                    logger.warning("Output underflow detected.")
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
                self.stop()  # Stop the stream if an error occurs
                raise RuntimeError("Callback error triggered stream termination.")

        try:
            self.stream = sd.OutputStream(
                samplerate=self.config.sample_rate,
                device=self.config.device,
                channels=1,
                blocksize=self.config.buffer_size,
                callback=wrapped_callback,
            )
            self.stream.start()
            logger.info(
                f"Audio stream started: {self.config.sample_rate}Hz, "
                f"buffer={self.config.buffer_size}"
            )
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.stream = None
            raise RuntimeError(f"Could not start audio stream: {e}")

    def stop(self) -> None:
        """Stop the audio stream and cleanup resources."""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
                logger.info("Audio stream stopped")
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
            finally:
                self.stream = None
