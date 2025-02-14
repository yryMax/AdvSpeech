import math

import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from librosa.filters import mel as librosa_mel_fn


class CheapTrick(nn.Module):
    def __init__(
        self,
        fs: int,
        q1: float = -0.15,
        f0_floor: float = 71.0,
        default_f0: float = 500.0,
    ):
        super().__init__()
        self.fs = fs
        self.q1 = q1
        self.f0_floor = f0_floor
        self.default_f0 = default_f0

    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, temporal_positions: torch.Tensor
    ) -> torch.Tensor:
        # 将 fs, f0_floor, default_f0 转为与 f0 相同 device 和 dtype 的 Tensor
        fs_t = torch.tensor(self.fs, dtype=f0.dtype, device=f0.device)
        f0_floor_t = torch.tensor(self.f0_floor, dtype=f0.dtype, device=f0.device)
        default_f0_t = torch.tensor(self.default_f0, dtype=f0.dtype, device=f0.device)

        # 计算 fft_size (需使用tensor运算）
        val = 3 * fs_t / f0_floor_t + 1.0
        fft_size = (2 ** torch.ceil(torch.log2(val))).int()

        f0_floor = 3.0 * fs_t / (3.0 * fft_size.float())
        f0 = torch.where(f0 < f0_floor, default_f0_t, f0)

        num_frames = f0.shape[0]
        spectrogram = torch.zeros(
            (fft_size // 2 + 1, num_frames), dtype=x.dtype, device=x.device
        )

        for i in range(num_frames):
            spectrogram[:, i] = self.estimate_one_slice(
                x, f0[i], temporal_positions[i], fft_size, self.q1
            )

        return spectrogram

    def estimate_one_slice(
        self,
        x: torch.Tensor,
        current_f0: torch.Tensor,
        current_position: torch.Tensor,
        fft_size: torch.Tensor,
        q1: float,
    ) -> torch.Tensor:
        fs_t = torch.tensor(self.fs, dtype=x.dtype, device=x.device)
        fft_size_val = int(fft_size.item())

        half_window_length = (1.5 * fs_t / current_f0 + 0.5).int()
        base_index = torch.arange(
            -half_window_length, half_window_length + 1, device=x.device, dtype=x.dtype
        )

        center = (current_position * fs_t + 0.501).int() + 1
        index = center + base_index
        index_clamped = torch.clamp(index, 1, x.shape[0]).long() - 1
        segment = x[index_clamped]

        time_axis = base_index / fs_t / 1.5
        window = 0.5 * torch.cos(torch.pi * time_axis * current_f0) + 0.5
        window = window / torch.sqrt(torch.sum(window**2))

        mean_seg = torch.mean(segment * window)
        mean_win = torch.mean(window)
        waveform = segment * window - window * (mean_seg / mean_win)

        waveform_fft = torch.fft.fft(waveform, n=fft_size_val)
        power_spectrum = waveform_fft.abs() ** 2
        frequency_axis = (
            torch.arange(0, fft_size_val, device=x.device, dtype=x.dtype)
            / fft_size_val
            * fs_t
        )

        cond = frequency_axis < (current_f0 + fs_t / fft_size_val)
        lfreq = frequency_axis[cond]
        lpspec = power_spectrum[cond]

        x_in = current_f0 - lfreq
        x_in_reversed, indices = torch.sort(x_in)
        y_in_reversed = lpspec[indices]

        cond2 = frequency_axis < current_f0
        query_points = frequency_axis[cond2]

        low_frequency_replica = self.interp1H_torch(
            x_in_reversed, y_in_reversed, query_points
        )

        power_spectrum[cond2] = power_spectrum[cond2] + low_frequency_replica

        half = fft_size_val // 2

        power_spectrum[half + 1 :] = power_spectrum[1:half].flip(0)

        double_frequency_axis = (
            torch.arange(2 * fft_size_val, device=x.device, dtype=x.dtype)
            / fft_size_val
            * fs_t
            - fs_t
        )
        double_spectrum = torch.cat([power_spectrum, power_spectrum], dim=0)

        delta = fs_t / fft_size_val
        double_segment = torch.cumsum(double_spectrum * delta, dim=0)

        center_frequency = (
            torch.arange(half + 1, device=x.device, dtype=x.dtype) / fft_size_val * fs_t
        )

        low_levels = self.interp1H_torch(
            double_frequency_axis + delta / 2,
            double_segment,
            center_frequency - current_f0 / 3,
        )
        high_levels = self.interp1H_torch(
            double_frequency_axis + delta / 2,
            double_segment,
            center_frequency + current_f0 / 3,
        )

        smoothed_spectrum = (high_levels - low_levels) * 1.5 / current_f0 + 1e-10

        tmp = torch.cat([smoothed_spectrum, smoothed_spectrum[1:-1].flip(0)], dim=0)

        quefrency_axis = (
            torch.arange(fft_size_val, device=x.device, dtype=x.dtype) / fs_t
        )

        smoothing_lifter = torch.empty_like(quefrency_axis)
        smoothing_lifter[0] = 1.0
        smoothing_lifter[1:] = torch.sin(torch.pi * current_f0 * quefrency_axis[1:]) / (
            torch.pi * current_f0 * quefrency_axis[1:]
        )

        compensation_lifter = (1 - 2 * q1) + 2 * q1 * torch.cos(
            2 * torch.pi * current_f0 * quefrency_axis
        )

        tandem_cepstrum = torch.fft.fft(torch.log(tmp))
        result_cepstrum = tandem_cepstrum * smoothing_lifter * compensation_lifter
        tmp_spectral_envelope = torch.exp(torch.real(torch.fft.ifft(result_cepstrum)))
        spectral_envelope = tmp_spectral_envelope[: half + 1]

        return spectral_envelope

    def interp1H_torch(
        self, x: torch.Tensor, y: torch.Tensor, xi: torch.Tensor
    ) -> torch.Tensor:
        delta_x = x[1] - x[0]
        xi_clamped = torch.clamp(xi, x[0], x[-1])
        xi_pos = (xi_clamped - x[0]) / delta_x
        xi_base = torch.floor(xi_pos).long()
        xi_fraction = xi_pos - xi_base

        delta_y = torch.empty_like(y)
        delta_y[:-1] = y[1:] - y[:-1]
        delta_y[-1] = torch.tensor(0.0, dtype=y.dtype, device=y.device)

        xi_base = torch.clamp(xi_base, 0, y.shape[0] - 1)
        yi = y[xi_base] + delta_y[xi_base] * xi_fraction
        return yi


class MelSpectrogramLayer(nn.Module):
    def __init__(
        self,
        n_fft: int = 1024,
        num_mels: int = 80,
        sampling_rate: int = 22050,
        hop_size: int = 256,
        win_size: int = 1024,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        center: bool = False,
        device: str = "cuda",
    ):
        super(MelSpectrogramLayer, self).__init__()
        self.device = device
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center

        self.register_buffer("hann_window", torch.hann_window(win_size).to(self.device))

        mel = librosa_mel_fn(
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.num_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel).float().to(self.device))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y = y.to(self.device)
        if torch.min(y) < -1.0 or torch.max(y) > 1.0:
            raise ValueError("Input tensor y must have values in the range [-1, 1]")

        y = F.pad(
            y.unsqueeze(1),
            (
                int((self.n_fft - self.hop_size) / 2),
                int((self.n_fft - self.hop_size) / 2),
            ),
            mode="reflect",
        ).squeeze(1)

        spec = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        spec = torch.sqrt(spec.real**2 + spec.imag**2 + 1e-9)

        mel_spec = torch.matmul(self.mel_basis, spec)

        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        return mel_spec
