import os
import re
import subprocess
from io import StringIO

import pandas as pd
import torch.nn.functional as F
from pesq import pesq
from resemblyzer import preprocess_wav
from resemblyzer import VoiceEncoder
from torchmetrics import Metric

from util import *


class SNRMetric(Metric):
    """
    https://colab.research.google.com/github/hrtlacek/SNR/blob/master/SNR.ipynb
    """

    is_differentiable = False
    higher_is_better = True

    def __init__(self):
        super().__init__()
        self.add_state(
            "snr_sum",
            default=torch.tensor(0.0, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_updates",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_of_squares",
            default=torch.tensor(0.0, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        assert pred.dim() == target.dim() == 2

        pred = pred.squeeze()
        target = target.squeeze()

        signal_energy = torch.mean(target**2)
        noise_energy = torch.mean((target - pred) ** 2)

        if noise_energy.item() == 0:
            snr_val = float("inf")
        elif signal_energy.item() == 0:
            snr_val = -float("inf")
        else:
            snr_val = 10.0 * torch.log10(signal_energy / noise_energy)
            print(snr_val)

        self.snr_sum += snr_val
        self.sum_of_squares += snr_val**2
        self.num_updates += 1

    def compute(self) -> torch.Tensor:
        if self.num_updates == 0:
            return torch.tensor(0.0, dtype=torch.float)

        mean_snr = self.snr_sum / self.num_updates
        std_snr = torch.sqrt(self.sum_of_squares / self.num_updates - mean_snr**2)

        return mean_snr, std_snr


class PESQMetric(Metric):
    is_differentiable = False
    higher_is_better = True

    def __init__(self, fs=16000, mode="wb"):
        super().__init__()
        if fs not in [8000, 16000]:
            raise ValueError("PESQ only supports 8k or 16k sample rate.")

        self.fs = fs
        self.mode = mode

        self.add_state(
            "pesq_sum",
            default=torch.tensor(0.0, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_updates",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_of_squares",
            default=torch.tensor(0.0, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        assert pred.dim() == target.dim() == 2

        pred = pred.squeeze()
        target = target.squeeze()

        pred, target = align_shape(pred, target)

        ref_wav = target.detach().cpu().numpy().astype("float32")
        deg_wav = pred.detach().cpu().numpy().astype("float32")

        try:
            pesq_score = pesq(self.fs, ref_wav, deg_wav, self.mode)
        except Exception as e:
            raise RuntimeError(f"PESQ calculation failed: {e}")

        self.pesq_sum += pesq_score
        self.sum_of_squares += pesq_score**2
        self.num_updates += 1

    def compute(self) -> torch.Tensor:
        if self.num_updates == 0:
            return torch.tensor(0.0, dtype=torch.float)
        mean = self.pesq_sum / self.num_updates
        std = torch.sqrt(self.sum_of_squares / self.num_updates - mean**2)
        return mean, std


class SECSMetric(Metric):
    is_differentiable = False
    higher_is_better = False

    def __init__(self, sr=16000):
        super().__init__()
        self.sr = sr
        self.add_state(
            "secs_sum",
            default=torch.tensor(0.0, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_updates",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_of_squares",
            default=torch.tensor(0.0, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.encoder = VoiceEncoder()

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        assert preds.dim() == target.dim() == 2

        preds = preds.squeeze(0)
        target = target.squeeze(0)
        preds, target = align_shape(preds, target)

        wav_pred = preds.detach().cpu().squeeze(0).numpy()
        wav_targ = target.detach().cpu().squeeze(0).numpy()

        embeds_pred = self.encoder.embed_utterance(
            preprocess_wav(wav_pred, source_sr=self.sr)
        )
        embeds_targ = self.encoder.embed_utterance(
            preprocess_wav(wav_targ, source_sr=self.sr)
        )

        cosim = np.dot(embeds_targ, embeds_pred) / (
            np.linalg.norm(embeds_targ) * np.linalg.norm(embeds_pred)
        )
        cosim_t = torch.tensor(cosim, device=preds.device, dtype=torch.float)

        self.secs_sum += 1 - cosim_t
        self.sum_of_squares += (1 - cosim_t) ** 2
        self.num_updates += 1

    def compute(self) -> torch.Tensor:
        if self.num_updates == 0:
            return torch.tensor(0.0, dtype=torch.float)
        mean = self.secs_sum / self.num_updates
        std = torch.sqrt(self.sum_of_squares / self.num_updates - mean**2)
        return mean, std


def mos_runner(path):
    if not os.path.exists(path):
        print("files are not saved, cannot calculate MOS")
        return None, None

    nisqa_dir = "external_repos/NISQA"

    cmd = [
        "conda",
        "run",
        "-n",
        "nisqa",
        "python",
        "run_predict.py",
        "--mode",
        "predict_dir",
        "--pretrained_model",
        "weights/nisqa.tar",
        "--data_dir",
        path,
    ]
    try:
        result = subprocess.run(
            cmd, cwd=nisqa_dir, capture_output=True, text=True, check=True
        )
        output = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] NISQA execution failed: {e}")
        print(e.stderr)
        return None, None

    return _parse_nisqa_output(output)


def _parse_nisqa_output(output):
    match = re.search(
        r"^\s*deg\s+mos_pred\s+noi_pred\s+dis_pred\s+col_pred\s+loud_pred",
        output,
        re.MULTILINE,
    )
    if not match:
        print("[ERROR] Failed to detect NISQA table in stdout.")
        return None

    table_start = match.start()
    table_content = output[table_start:]

    df = pd.read_csv(StringIO(table_content), delim_whitespace=True)
    mos_mean = df["mos_pred"].mean()
    mos_std = df["mos_pred"].std()

    return mos_mean, mos_std


if __name__ == "__main__":
    original = load_wav("../audios/cn_sample/original.wav", 16000)
    distorted = load_wav("../audios/cn_sample/ry_adv.wav", 16000)
    original = original[:, : distorted.size(1)]
    distorted = distorted[:, : original.size(1)]
    snr_metric = SNRMetric()
    snr_metric.update(distorted, original)
    print(snr_metric.compute())
    pesq_metric = PESQMetric()
    pesq_metric.update(distorted, original)
    print(pesq_metric.compute())
    secs_metric = SECSMetric()
    secs_metric.update(distorted, original)
    print(secs_metric.compute())
