import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import torch

def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate >= target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


def draw_mel_spectrogram(mel_tensor, title='Mel Spectrogram'):
    mel_np = mel_tensor.squeeze().cpu().numpy()

    plt.figure(figsize=(10, 6), dpi=200)
    plt.imshow(mel_np.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Frequency Channels')
    plt.title(title)
    plt.tight_layout()

    plt.show()


def audiosave(audio, name):
    torchaudio.save('./adv_example/' + name, audio, 22050)


def draw_envelope(envelope_tensor):
    plt.imshow(20 * np.log10(envelope_tensor + 1e-6), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.title("Spectral Envelope")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()

def tensor_normalize(x_tensor):
    x = torch.log10(x_tensor + 1e-6)
    x = (x - x.min()) / (x.max() - x.min())
    return x



def align_shape(audiotensor_1: torch.Tensor, audiotensor_2: torch.Tensor):
    assert audiotensor_1.dim() == audiotensor_2.dim()
    min_length = min(audiotensor_1.shape[-1], audiotensor_2.shape[-1])
    audiotensor_1 = audiotensor_1[..., :min_length]
    audiotensor_2 = audiotensor_2[..., :min_length]
    return audiotensor_1, audiotensor_2