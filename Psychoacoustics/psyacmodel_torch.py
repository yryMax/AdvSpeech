# Programs to implement a psycho-acoustic model
# Using a matrix for the spreading function (faster)
# Gerald Schuller, Nov. 2016
# torch version from Renato Profeta, Feb 2024
import torch


def hz2bark_torch(f, device=torch.device("cuda")):
    """
    Usage: Bark=hz2bark_torch(f)
    f    : (float or Tensor)  Frequencies in Hz.
    Returns:
        Brk : (Tensor) Bark scaled values.
    """
    if device is None:
        device = torch.device("cpu")
    if not torch.is_tensor(f):
        f = torch.tensor(f, device=device, dtype=torch.float32)
    else:
        f = f.to(device=device, dtype=torch.float32)

    Brk = 6.0 * torch.arcsinh(f / 600.0)
    return Brk


def bark2hz_torch(Brk, device=torch.device("cuda")):
    """
    Usage: Hz=bark2hz_torch(Brk)
    Brk   : (float or Tensor) Bark scaled values.
    Returns:
        Fhz : (Tensor) frequencies in Hz.
    """
    if device is None:
        device = torch.device("cpu")
    if not torch.is_tensor(Brk):
        Brk = torch.tensor(Brk, device=device, dtype=torch.float32)
    else:
        Brk = Brk.to(device=device, dtype=torch.float32)

    Fhz = 600.0 * torch.sinh(Brk / 6.0)
    return Fhz


def f_SP_dB_torch(maxfreq, nfilts, device=torch.device("cuda")):
    """
    usage: spreadingfunctionBarkdB = f_SP_dB_torch(maxfreq, nfilts)
    computes the spreading function prototype in the Bark scale.
    Arguments:
        maxfreq : half the sampling frequency
        nfilts  : number of subbands in the Bark domain, e.g. 64
    Returns:
        spreadingfunctionBarkdB : (Tensor) shape (2*nfilts,)
    """
    if device is None:
        device = torch.device("cpu")
    # 将 maxfreq 转为 Bark
    maxbark = hz2bark_torch(maxfreq, device=device)

    # 构造 spreadingfunctionBarkdB
    spreadingfunctionBarkdB = torch.zeros(2 * nfilts, device=device)
    # 前 nfilts: 低频侧
    spreadingfunctionBarkdB[0:nfilts] = (
        torch.linspace(-maxbark * 27, -8, nfilts, device=device) - 23.5
    )
    # 后 nfilts: 高频侧
    spreadingfunctionBarkdB[nfilts : 2 * nfilts] = (
        torch.linspace(0, -maxbark * 12.0, nfilts, device=device) - 23.5
    )

    return spreadingfunctionBarkdB


def spreadingfunctionmat_torch(
    spreadingfunctionBarkdB, alpha, nfilts, device=torch.device("cuda")
):
    """
    Turns the spreading prototype function (in dB) into a matrix of shifted versions.
    Convert from dB to "voltage" and include alpha exponent.
    nfilts: Number of subbands in the Bark domain, e.g. 64
    """
    if device is None:
        device = torch.device("cpu")

    spreadingfunctionBarkdB = spreadingfunctionBarkdB.to(device=device)
    # 转成 "voltage" 并加上 alpha
    spreadingfunctionBarkVoltage = 10.0 ** (spreadingfunctionBarkdB / 20.0 * alpha)

    # 构造输出矩阵
    spreadingfuncmatrix = torch.zeros((nfilts, nfilts), device=device)
    for k in range(nfilts):
        # 对应 shifting
        spreadingfuncmatrix[k, :] = spreadingfunctionBarkVoltage[
            (nfilts - k) : (2 * nfilts - k)
        ]

    return spreadingfuncmatrix


def maskingThresholdBark_torch(
    mXbark, spreadingfuncmatrix, alpha, fs, nfilts, device=torch.device("cuda")
):
    """
    Computes the masking threshold on the Bark scale with non-linear superposition.
    mXbark: magnitude of spectrum on the Bark scale
    spreadingfuncmatrix: from spreadingfunctionmat_torch
    alpha: exponent for non-linear superposition (e.g. 0.6 ~ 0.8)
    fs: sampling frequency
    nfilts: number of Bark subbands
    Returns:
        mTbark: the resulting Masking Threshold on the Bark scale
    """
    if device is None:
        device = torch.device("cpu")

    mXbark = mXbark.to(device=device)
    spreadingfuncmatrix = spreadingfuncmatrix.to(device=device)

    # 非线性叠加
    mTbark = torch.matmul(mXbark**alpha, spreadingfuncmatrix**alpha)
    mTbark = mTbark ** (1.0 / alpha)

    # Threshold in quiet:
    maxfreq = fs / 2.0
    maxbark = hz2bark_torch(maxfreq, device=device)
    step_bark = maxbark / (nfilts - 1)
    barks = torch.arange(0, nfilts, device=device, dtype=torch.float32) * step_bark
    # 转回 Hz
    f = bark2hz_torch(barks, device=device) + 1e-6

    # 临界带听阈（dB）
    LTQ = (
        3.64 * (f / 1000.0) ** -0.8
        - 6.5 * torch.exp(-0.6 * ((f / 1000.0) - 3.3) ** 2.0)
        + 1e-3 * (f / 1000.0) ** 4.0
    )
    # clip到[-20,120]区间
    LTQ = torch.clip(LTQ, -20, 120)

    # 转成线性刻度
    a = mTbark
    b = 10.0 ** ((LTQ - 60) / 20)
    mTbark = torch.max(a, b)  # 与安静阈值取更大的那个

    return mTbark


def mapping2barkmat_torch(fs, nfilts, nfft, device=torch.device("cuda")):
    """
    Constructs a mapping matrix W with 1's for each Bark subband, 0's otherwise.
    usage: W = mapping2barkmat_torch(fs, nfilts, nfft)
    fs: sampling frequency
    nfilts: number of subbands in Bark domain
    nfft: FFT size
    Returns:
        W: (nfilts, nfft)  (Tensor)
    """
    if device is None:
        device = torch.device("cpu")

    maxbark = hz2bark_torch(fs / 2, device=device)
    nfreqs = int(nfft / 2)
    step_bark = maxbark / (nfilts - 1)

    # 频率到 Bark
    freqs = torch.linspace(0, nfreqs, steps=nfreqs + 1, device=device) * (fs / nfft)
    binbark = hz2bark_torch(freqs, device=device)

    W = torch.zeros((nfilts, nfft), device=device)
    for i in range(nfilts):
        # 把频率轴上(0..nfft/2)对应的那部分，映射到 i Bark
        # 用 round(binbark/step_bark)==i 来判断属于哪个 Bark 子带
        W[i, 0 : nfreqs + 1] = torch.round(binbark / step_bark) == i

    return W


def mapping2bark_torch(mX, W, nfft, device=torch.device("cuda")):
    """
    Maps (warps) magnitude spectrum vector mX from DFT to the Bark scale.
    mX: magnitude spectrum from FFT, shape ~ (nfft/2+1,)
    W: mapping matrix from mapping2barkmat_torch
    nfft: FFT size
    Returns: mXbark, shape ~ (nfilts,)
    """
    if device is None:
        device = torch.device("cpu")

    mX = mX.to(device=device)
    W = W.to(device=device)

    nfreqs = int(nfft / 2)
    # sum powers in each Bark band, then take sqrt -> "voltage"
    # note: mX[:nfreqs+1]
    power_spectrum = (mX[: nfreqs + 1]) ** 2.0

    # 矩阵乘法： (1 x (nfreqs+1)) x ((nfreqs+1) x nfilts) => 1 x nfilts
    # 但 W 是 (nfilts x nfft)，需要转置再切片: W[:, :nfreqs+1].T -> ((nfreqs+1) x nfilts)
    mXbark = torch.matmul(power_spectrum, W[:, : nfreqs + 1].T)
    mXbark = mXbark**0.5

    return mXbark


def mappingfrombarkmat_torch(W, nfft, device=torch.device("cuda")):
    """
    Constructs inverse mapping matrix W_inv from W for mapping back from Bark scale.
    usage: W_inv = mappingfrombarkmat_torch(W, nfft)
    W: (nfilts, nfft)
    nfft: FFT size
    returns: W_inv: shape ~ ((nfft/2+1), nfilts)
    """
    if device is None:
        device = torch.device("cpu")

    W = W.to(device=device)
    nfreqs = int(nfft / 2)
    # 计算按行求和(每个 Bark 通道上多少FFT频点)，再取 sqrt 的逆
    denom = (torch.sum(W, dim=1) + 1e-6) ** 0.5
    diag_mat = torch.diag(1.0 / denom).to(device=device)

    # W_inv shape: ((nfft/2+1), nfilts)
    W_inv = torch.matmul(diag_mat, W[:, : nfreqs + 1]).T
    return W_inv


def mappingfrombark_torch(mTbark, W_inv, nfft, device=torch.device("cuda")):
    """
    Maps (warps) a magnitude spectrum vector mTbark in the Bark scale
    back to the linear frequency domain.
    usage: mT = mappingfrombark_torch(mTbark, W_inv, nfft)
    mTbark : (nfilts,) Masking threshold in the Bark domain
    W_inv  : inverse mapping matrix from mappingfrombarkmat_torch
    nfft   : FFT size
    returns:
        mT   : shape (nfft/2+1,) Masking threshold in the linear scale
    """
    if device is None:
        device = torch.device("cpu")

    mTbark = mTbark.to(device=device)
    W_inv = W_inv.to(device=device)

    nfreqs = int(nfft / 2)
    # (1 x nfilts) x (nfilts x (nfreqs+1)) => 1 x (nfreqs+1)
    mT = torch.matmul(mTbark, W_inv[:, :nfreqs].T.float())
    return mT


if __name__ == "__main__":
    # testing:
    import matplotlib.pyplot as plt

    fs = 32000  # sampling frequency of audio signal
    maxfreq = fs / 2
    alpha = 0.8  # Exponent for non-linear superposition of spreading functions
    nfilts = 64  # number of subbands in the bark domain
    nfft = 2048  # number of fft subbands

    W = mapping2barkmat_torch(fs, nfilts, nfft)
    plt.imshow(W[:, :256], cmap="Blues")
    plt.title("Matrix W for Uniform to Bark Mapping as Image")
    plt.xlabel("Uniform Subbands")
    plt.ylabel("Bark Subbands")
    plt.show()

    W_inv = mappingfrombarkmat_torch(W, nfft)
    plt.imshow(W_inv[:256, :], cmap="Blues")
    plt.title("Matrix W_inv for Bark to Uniform Mapping as Image")
    plt.xlabel("Bark Subbands")
    plt.ylabel("Uniform Subbands")
    plt.show()

    spreadingfunctionBarkdB = f_SP_dB_torch(maxfreq, nfilts)
    # x-axis: maxbark Bark in nfilts steps:
    maxbark = hz2bark_torch(maxfreq)
    print("maxfreq=", maxfreq, "maxbark=", maxbark)
    bark = torch.linspace(0, maxbark, nfilts)
    # The prototype over "nfilt" bands or 22 Bark, its center
    # shifted down to 22-26/nfilts*22=13 Bark:
    plt.plot(bark, spreadingfunctionBarkdB[26 : (26 + nfilts)])
    plt.axis([6, 23, -100, 0])
    plt.xlabel("Bark")
    plt.ylabel("dB")
    plt.title("Spreading Function")
    plt.show()

    spreadingfuncmatrix = spreadingfunctionmat_torch(
        spreadingfunctionBarkdB, alpha, nfilts
    )
    plt.imshow(spreadingfuncmatrix)
    plt.title("Matrix spreadingfuncmatrix as Image")
    plt.xlabel("Bark Domain Subbands")
    plt.ylabel("Bark Domain Subbands")
    plt.show()

    # -Testing-----------------------------------------
    # A test magnitude spectrum:
    # White noise:
    x = torch.randn(32000) * 1000

    mX = torch.abs(torch.fft.fft(x[0:2048], norm="ortho"))[0:1025]
    mXbark = mapping2bark_torch(mX, W, nfft)
    # Compute the masking threshold in the Bark domain:
    mTbark = maskingThresholdBark_torch(mXbark, spreadingfuncmatrix, alpha, fs, nfilts)
    # Massking threshold in the original frequency domain
    mT = mappingfrombark_torch(mTbark, W_inv, nfft)
    plt.plot(20 * torch.log10(mX + 1e-3))
    plt.plot(20 * torch.log10(mT + 1e-3))
    plt.title("Masking Theshold for White Noise")
    plt.legend(("Magnitude Spectrum White Noise", "Masking Threshold"))
    plt.xlabel("FFT subband")
    plt.ylabel("Magnitude ('dB')")
    plt.show()
    # ----------------------------------------------
    # A test magnitude spectrum, an idealized tone in one subband:
    # tone at FFT band 200:
    x = torch.sin(2 * torch.pi / nfft * 200 * torch.arange(32000)) * 1000

    mX = torch.abs(torch.fft.fft(x[0:2048], norm="ortho"))[0:1025]
    # Compute the masking threshold in the Bark domain:
    mXbark = mapping2bark_torch(mX, W, nfft)
    mTbark = maskingThresholdBark_torch(mXbark, spreadingfuncmatrix, alpha, fs, nfilts)
    mT = mappingfrombark_torch(mTbark, W_inv, nfft)
    plt.plot(20 * torch.log10(mT + 1e-3))
    plt.title("Masking Theshold for a Tone")
    plt.plot(20 * torch.log10(mX + 1e-3))
    plt.legend(("Masking Trheshold", "Magnitude Spectrum Tone"))
    plt.xlabel("FFT subband")
    plt.ylabel("dB")
    plt.show()

    # stft, norm='ortho':
    # import scipy.signal
    # f,t,y=scipy.signal.stft(x,fs=32000,nperseg=2048)
    # make it orthonormal for Parsevals Theorem:
    # Hann window power per sample: 0.375
    # y=y*sqrt(2048/2)/2/0.375
    # plot(y[:,1])
    # plot(mX)

    """
  y=zeros((1025,3))
  y[0,0]=1
  t,x=scipy.signal.istft(y,window='boxcar')
  plot(x)
  #yields rectangle with amplitude 1/2, for orthonormality it would be sqrt(2/N) with overlap,
  #hence we need a factor sqrt(2/N)*2 for the synthesis, and sqrt(N/2)/2 for the analysis
  #for othogonality.
  #Hence it needs factor sqrt(N/2)/2/windowpowerpersample, hence for Hann Window:
  #y=y*sqrt(2048/2)/2/0.375
  """
