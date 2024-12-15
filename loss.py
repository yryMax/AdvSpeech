import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math


class SSIMLossLayer(nn.Module):
    def __init__(self, reference, channel: int = 1, window_size: int = 11, size_average: bool = True,
                 device: str = 'cuda'):
        super(SSIMLossLayer, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.device = device

        if not isinstance(reference, torch.Tensor):
            raise ValueError("Reference must be a PyTorch tensor.")
        self.reference = reference.to(self.device).unsqueeze(0)

        window = self.create_window(window_size, 1.5, channel)
        self.register_buffer('window', window.to(self.device))

    def create_window(self, window_size, sigma, channel):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        gauss /= gauss.sum()
        _2D_window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        _2D_window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return _2D_window

    def forward(self, mel_input):

        mel_input = mel_input.to(self.device).unsqueeze(0)
        mel_input = mel_input[:, :, :self.reference.shape[2]]

        # print(mel_input.shape)
        # print(self.reference.shape)

        if mel_input.shape != self.reference.shape:
            raise ValueError("Input tensors must have the same shape!")

        mu1 = F.conv2d(mel_input, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(self.reference, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(mel_input * mel_input, self.window, padding=self.window_size // 2,
                             groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(self.reference * self.reference, self.window, padding=self.window_size // 2,
                             groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(mel_input * self.reference, self.window, padding=self.window_size // 2,
                           groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return torch.clamp((1 - ssim_map.mean()) / 2, 0, 1)
        else:
            return torch.clamp((1 - ssim_map) / 2, 0, 1)


class MFCCLoss(nn.Module):
    def __init__(self, sample_rate=22050, n_mfcc=20, n_mels=80, n_fft=1024, hop_length=256):
        super(MFCCLoss, self).__init__()
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        )
        self.log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)

    def forward(self, ref_audio: torch.Tensor, syn_audio: torch.Tensor) -> torch.Tensor:
        ref_mfcc = self.mfcc_transform(ref_audio)
        syn_mfcc = self.mfcc_transform(syn_audio)

        assert ref_mfcc.shape == syn_mfcc.shape, "The shape of the reference and synthesized MFCCs must be the same."

        diff = ref_mfcc - syn_mfcc

        diff_per_frame = torch.sqrt(torch.sum(diff ** 2, dim=1))

        distance = torch.mean(diff_per_frame)

        mcd_val = self.log_spec_dB_const * distance

        norm_val = mcd_val / (mcd_val + 1)
        return norm_val


class SpectralFlatnessLoss(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, eps=1e-8):
        super(SpectralFlatnessLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self.window = None

    def forward(self, ref_audio: torch.Tensor, adv_audio: torch.Tensor) -> torch.Tensor:
        if ref_audio.dim() == 1:
            ref_audio = ref_audio.unsqueeze(0)
        if adv_audio.dim() == 1:
            adv_audio = adv_audio.unsqueeze(0)

        if self.window is None or self.window.device != adv_audio.device:
            self.window = torch.hann_window(self.n_fft, device=adv_audio.device)

        noise = adv_audio - ref_audio

        noise_stft = torch.stft(noise, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=self.window, return_complex=True)

        noise_mag = noise_stft.abs() + self.eps

        noise_log_mean = torch.mean(torch.log(noise_mag), dim=1)  # (batch, frames)
        noise_geo_mean = torch.exp(noise_log_mean)

        noise_arith_mean = torch.mean(noise_mag, dim=1)

        noise_sfm = noise_geo_mean / noise_arith_mean  # (batch, frames)
        noise_sfm_mean = torch.mean(noise_sfm, dim=1)  # (batch,)


        loss = torch.mean(torch.abs(noise_sfm_mean - 1.0))

        return loss

class NormalShapeLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(NormalShapeLoss, self).__init__()
        self.eps = eps

    def forward(self, ref_audio: torch.Tensor, adv_audio: torch.Tensor) -> torch.Tensor:
        noise = adv_audio - ref_audio
        if noise.dim() == 1:
            noise = noise.unsqueeze(0)  # (1, time) if needed

        mean = noise.mean()
        std = noise.std(unbiased=False) + self.eps
        noise_normalized = (noise - mean) / std

        loss = (noise_normalized.pow(2).mean() - 1.0).pow(2)
        return loss


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY