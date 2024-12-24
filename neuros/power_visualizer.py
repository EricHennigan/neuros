import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class PowerVisualizer:
    def __init__(self, num_channels: int, sample_rate: int, history_length: int = 50):
        self.num_channels = num_channels
        self.sampling_rate = sample_rate
        self.history_length = history_length
        self.histories = {
            'alpha': [deque(maxlen=history_length) for _ in range(num_channels)],
            'theta': [deque(maxlen=history_length) for _ in range(num_channels)],
            'beta': [deque(maxlen=history_length) for _ in range(num_channels)],
            'ratios': [deque(maxlen=history_length) for _ in range(num_channels)]
        }
        self.time_history = deque(maxlen=history_length)

        # Initialize plot elements
        self.fig = None
        self.power_lines = None
        self.ax_powers = None
        self.spec_data = None
        self.spec_img = None
        self.bars = None
        self.ax_bars = None
        self.ax_spec = None
        self.ratio_lines = None
        self.ax_ratios = None

        plt.ion()
        self.setup_plots()

    def setup_plots(self):
        """Create multi-panel visualization."""
        self.fig = plt.figure(figsize=(15, 10))

        # Time series of band powers
        self.ax_powers = self.fig.add_subplot(221)
        # Only plot alpha for each channel for cleaner visualization
        self.power_lines = {
            'alpha': [
                self.ax_powers.plot([], [], label=f'Ch{i} alpha')[0]
                for i in range(self.num_channels)
            ]
        }
        self.ax_powers.set_ylabel('Alpha Power')
        self.ax_powers.set_ylabel('Band Power')
        self.ax_powers.set_xlabel('Time (s)')
        self.ax_powers.grid(True)
        self.ax_powers.legend()

        # Band ratios
        self.ax_ratios = self.fig.add_subplot(222)
        self.ratio_lines = [
            self.ax_ratios.plot([], [], label=f'Ch{i} α/θ')[0]
            for i in range(self.num_channels)
        ]
        self.ax_ratios.set_ylabel('Alpha/Theta Ratio')
        self.ax_ratios.set_xlabel('Time (s)')
        self.ax_ratios.grid(True)
        self.ax_ratios.legend()

        # Real-time spectrogram (for first channel)
        self.ax_spec = self.fig.add_subplot(223)

        # Bar plot of current band powers
        self.ax_bars = self.fig.add_subplot(224)
        x = np.arange(self.num_channels)
        self.bars = self.ax_bars.bar(
            x,
            np.zeros(self.num_channels),
            tick_label=[f'Ch{i}' for i in range(self.num_channels)]
        )
        self.ax_bars.set_ylabel('Current Alpha Power')
        self.ax_bars.set_ylim(0, 1)  # Force positive range

        plt.tight_layout()

    def _initialize_spectrogram(self, psd_size):
        """Initialize spectrogram buffer with correct size"""
        self.spec_data = np.zeros((psd_size, self.history_length))

        # Calculate frequency bins
        nyquist = sampling_rate / 2
        freqs = np.linspace(0, nyquist, psd_size)

        self.spec_img = self.ax_spec.imshow(
            self.spec_data,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            cmap='viridis',
            extent=[0, self.history_length, freqs[0], freqs[-1]],
            vmin=0,  # Force positive range for power
            vmax=1
        )
        self.ax_spec.set_ylabel('Frequency (Hz)')
        self.ax_spec.set_xlabel('Time (s)')
        plt.colorbar(self.spec_img, ax=self.ax_spec, label='Power')

    def update(self, data: dict, timestamp: float):
        """Update visualization with new data."""
        self.time_history.append(timestamp)
        t = list(self.time_history)

        # Update band power lines
        for band in ['alpha', 'theta', 'beta']:
            for i in range(self.num_channels):
                self.histories[band][i].append(data['band_powers'][i][band])
                self.power_lines[band][i].set_data(t, list(self.histories[band][i]))

        # Update ratio lines
        for i in range(self.num_channels):
            self.histories['ratios'][i].append(data['ratios'][i]['alpha/theta'])
            self.ratio_lines[i].set_data(t, list(self.histories['ratios'][i]))

        # Update spectrogram
        if 'spectral' in data and len(data['spectral']) > 0:
            psd = data['spectral'][0]['psd']

            # Initialize spectrogram buffer if needed
            if self.spec_data is None:
                self._initialize_spectrogram(len(psd))

            # Update spectrogram
            self.spec_data = np.roll(self.spec_data, -1, axis=1)
            self.spec_data[:, -1] = psd
            self.spec_img.set_array(self.spec_data)

        # Update bar plot
        for i, bar in enumerate(self.bars):
            bar.set_height(data['band_powers'][i]['alpha'])

        # Update axis limits
        for ax in [self.ax_powers, self.ax_ratios]:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
