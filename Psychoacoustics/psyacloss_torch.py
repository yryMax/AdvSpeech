# Psycho-acoustic threshold function
# Gerald Schuller, September 2023
# torch version from Renato Profeta and Gerald Schuller, Feb 2024
import torch

from .psyacmodel_torch import *


def psyacthresh_torch(ys, fs):
    """
    Input:
        ys: 2D array (Tensor) of sound STFT from a mono signal, shape = (N+1, M)
        fs: sampling frequency in samples per second
    Returns:
        mT: the masking threshold in (N+1) subbands for the M blocks, shape = (N+1, M)
    """

    device = ys.device

    maxfreq = fs / 2
    alpha = 0.8
    nfilts = 64
    M = ys.shape[1]
    N = ys.shape[0] - 1
    nfft = 2 * N

    W = mapping2barkmat_torch(fs, nfilts, nfft).to(device)
    W_inv = mappingfrombarkmat_torch(W, nfft).to(device)
    spreadingfunctionBarkdB = f_SP_dB_torch(maxfreq, nfilts).to(device)
    spreadingfuncmatrix = spreadingfunctionmat_torch(
        spreadingfunctionBarkdB, alpha, nfilts
    ).to(device)

    mT = torch.zeros((N + 1, M), device=device)

    for m in range(M):
        mX = torch.abs(ys[:, m])

        mXbark = mapping2bark_torch(mX, W, nfft)

        mTbark = maskingThresholdBark_torch(
            mXbark, spreadingfuncmatrix, alpha, fs, nfilts
        )

        mT[:, m] = mappingfrombark_torch(mTbark, W_inv, nfft)

    return mT


def percloss(orig, modified, fs):
    """
    Computes the perceptually weighted distance between the original (orig)
    and modified audio signals, with sampling rate fs. The psycho-acoustic
    threshold is computed from orig, hence it is not commutative.

    Returns:
        ploss: the perceptual loss value (the mean squared difference of
               the two spectra, normalized to the masking threshold of the orig).
    """

    device = orig.device
    nfft = 2048
    N = nfft // 2

    window = torch.hann_window(2 * N, device=device)

    if len(orig.shape) == 2:
        chan = orig.shape[1]
        for c in range(chan):
            origys = torch.stft(
                orig[:, c],
                n_fft=2 * N,
                hop_length=N,
                window=window,
                return_complex=True,
                normalized=True,
            )
            mT0 = psyacthresh_torch(origys, fs).to(device)

            if c == 0:
                rows, cols = mT0.shape
                mT = torch.zeros((rows, chan, cols), device=device)
                mT[:, 0, :] = mT0
            else:
                mT[:, c, :] = mT0
    else:
        origys = torch.stft(
            orig,
            n_fft=2 * N,
            hop_length=N,
            window=window,
            return_complex=True,
            normalized=True,
        )
        mT = psyacthresh_torch(origys, fs).to(device)

    modifiedys = torch.stft(
        modified,
        n_fft=2 * N,
        hop_length=N,
        window=window,
        return_complex=True,
        normalized=True,
    )

    normdiffspec = torch.abs((origys - modifiedys) / mT)

    ploss = torch.mean(normdiffspec**2)
    return ploss


if __name__ == "__main__":  # testing
    import scipy.io.wavfile as wav
    import scipy.signal
    import numpy as np
    import matplotlib.pyplot as plt
    import sound
    import os

    fs, snd = wav.read(r"../adv_speech/6319_1.wav")
    plt.plot(snd)
    plt.title("The original sound")
    plt.show()

    nfft = 2048  # number of fft subbands
    N = nfft // 2

    print("snd.shape=", snd.shape)
    f, t, ys = scipy.signal.stft(snd, fs=2 * np.pi, nperseg=2 * N)
    # scaling for the application of the
    # resulting masking threshold to MDCT subbands:
    ys *= np.sqrt(2 * N / 2) / 2 / 0.375

    print("fs=", fs)
    ys = torch.from_numpy(ys)
    mT = psyacthresh_torch(ys, fs)

    print("mT.shape=", mT.shape)
    plt.plot(20 * np.log10(np.abs(ys[:, 400]) + 1e-6))
    plt.plot(20 * np.log10(mT[:, 400] + 1e-6))
    plt.legend(("Original spectrum", "Masking threshold"))
    plt.title("Spectrum over bins")

    plt.figure()
    plt.imshow(20 * np.log10(np.abs(ys) + 1e-6))
    plt.title("Spectrogram of Original")
    plt.show()

    # Audio signal with uniform quantization and de-quantization
    snd = torch.from_numpy(snd[:, 0]).float()
    snd_quant = (torch.round(snd / 10000)) * 10000

    print("\nThe quantized signal:")
    sound.sound(np.array(snd_quant), fs)

    ploss = percloss(snd, snd_quant, fs)

    # version AAC encoded and decoded:
    os.system("ffmpeg -y -i fantasy-orchestra.wav -b:a 64k fantasy-orchestra64k.aac")
    os.system("ffmpeg -y -i fantasy-orchestra64k.aac fantasy-orchestradec_aac.wav")
    fs, snd_aac = wav.read(r"./fantasy-orchestradec_aac.wav")

    print("\nThe AAC encoded/Decoded Signal:")
    sound.sound(np.array(snd_aac), fs)

    print("\n\npsyco-acoustic loss to quantized signal=", ploss)

    snd_aac = torch.from_numpy(snd_aac[:, 0]).float()

    minlength = min(snd.shape[0], snd_aac.shape[0])
    # print("\n\nminlength=", minlength)
    delay = 120  # aac delay in samples

    ploss_aac = percloss(snd[:minlength], snd_aac[delay : minlength + delay], fs)
    print("\n\npsyco-acoustic loss to aac enc/dec signal=", ploss_aac)
