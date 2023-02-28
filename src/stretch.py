import soundfile as sf
import numpy as np
from scipy import signal


class Stretch:
    def __init__(self, path: str):
        self.stretched_wave = None
        with open(path, 'rb') as f:
            self.wave, self.sr = sf.read(f)

    def stretch(self,
                rate: float = 1,
                bins: int = 2048,
                overlap: float = 0.75):
        fourier = signal.stft(self.wave, nperseg=bins, noverlap=int(bins * overlap))[2]
        voc = self.__phase_vocoder(fourier, rate=rate, hop_length=int(bins * overlap))
        y_stretch = signal.istft(voc, nperseg=bins, noverlap=int(bins * overlap))
        self.stretched_wave = y_stretch[1]

    def write(self, out_path: str):
        with open(out_path, 'wb') as f:
            sf.write(f, self.stretched_wave, self.sr)

    @staticmethod
    def __phase_vocoder(fourier: np.ndarray,
                        rate: float,
                        hop_length: int = None) -> np.ndarray:
        time_steps = np.arange(0, fourier.shape[-1], rate, dtype=np.float64)
        shape = list(fourier.shape)
        shape[-1] = len(time_steps)
        d_stretch = np.zeros_like(fourier, shape=shape)
        phi_advance = np.linspace(0, np.pi * hop_length, fourier.shape[-2])
        phase_acc = np.angle(fourier[:, 0])
        padding = [(0, 0) for _ in fourier.shape]
        padding[-1] = (0, 2)
        fourier = np.pad(fourier, padding, mode="constant")

        for t, step in enumerate(time_steps):
            columns = fourier[:, int(step): int(step + 2)]
            alpha = np.mod(step, 1.0)
            mag = (1.0 - alpha) * np.abs(columns[:, 0]) + alpha * np.abs(columns[:, 1])
            d_stretch[:, t] = (np.cos(phase_acc) + 1j * np.sin(phase_acc)) * mag
            phase = np.angle(columns[:, 1]) - np.angle(columns[:, 0]) - phi_advance
            phase = phase - 2.0 * np.pi * np.round(phase / (2.0 * np.pi))
            phase_acc += phi_advance + phase

        return d_stretch
