import torch
import torch.nn.functional as F
from torchmetrics import Metric
from util import *
from pesq import pesq
from resemblyzer import preprocess_wav, VoiceEncoder


class SNRMetric(Metric):

    is_differentiable = False
    higher_is_better = True

    def __init__(self):
        super().__init__()
        self.add_state("snr_sum", default=torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("num_updates", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")


    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        print(pred.shape, target.shape)
        assert pred.dim() == target.dim() == 2

        pred = pred.squeeze()
        target = target.squeeze()
        pred, target = align_shape(pred, target)

        signal_energy = F.mse_loss(target, torch.zeros_like(target), reduction='mean')
        noise_energy  = F.mse_loss(target, pred, reduction='mean')

        if noise_energy.item() == 0:
            snr_val = float('inf')
        else:
            snr_val = 10.0 * torch.log10(signal_energy / noise_energy).item()

        self.snr_sum += snr_val
        self.num_updates += 1

    def compute(self) -> torch.Tensor:
        if self.num_updates == 0:
            return torch.tensor(0., dtype=torch.float)
        return self.snr_sum / self.num_updates


class PESQMetric(Metric):
    is_differentiable = False
    higher_is_better = True

    def __init__(self, fs=16000, mode='wb'):
        super().__init__()
        if fs not in [8000, 16000]:
            raise ValueError("PESQ only supports 8k or 16k sample rate.")

        self.fs = fs
        self.mode = mode

        self.add_state("pesq_sum", default=torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("num_updates", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:

        assert pred.dim() == target.dim() == 2

        pred = pred.squeeze()
        target = target.squeeze()

        pred, target = align_shape(pred, target)

        ref_wav = target.detach().cpu().numpy().astype('float32')
        deg_wav = pred.detach().cpu().numpy().astype('float32')

        try:
            pesq_score = pesq(self.fs, ref_wav, deg_wav, self.mode)
        except Exception as e:
            raise RuntimeError(f"PESQ calculation failed: {e}")

        self.pesq_sum += pesq_score
        self.num_updates += 1

    def compute(self) -> torch.Tensor:
        if self.num_updates == 0:
            return torch.tensor(0., dtype=torch.float)
        return self.pesq_sum / self.num_updates


class SECSMetric(Metric):

    is_differentiable = False
    higher_is_better = False

    def __init__(self, sr=16000):

        super().__init__()
        self.sr = sr
        self.add_state("secs_sum", default=torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("num_updates", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.encoder = VoiceEncoder()

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:

        assert preds.dim() == target.dim() == 2

        preds = preds.squeeze(0)
        target = target.squeeze(0)
        preds, target = align_shape(preds, target)

        wav_pred = preds.detach().cpu().squeeze(0).numpy()
        wav_targ = target.detach().cpu().squeeze(0).numpy()


        embeds_pred = self.encoder.embed_utterance(preprocess_wav(wav_pred, source_sr=self.sr))
        embeds_targ = self.encoder.embed_utterance(preprocess_wav(wav_targ, source_sr=self.sr))

        cosim = np.dot(embeds_targ, embeds_pred) / (np.linalg.norm(embeds_targ) * np.linalg.norm(embeds_pred))
        cosim_t = torch.tensor(cosim, device=preds.device, dtype=torch.float)
        self.secs_sum += 1 - cosim_t
        self.num_updates += 1

    def compute(self) -> torch.Tensor:
        if self.num_updates == 0:
            return torch.tensor(0., dtype=torch.float)
        return self.secs_sum / self.num_updates


if __name__ == '__main__':
    original = load_wav('../audios/cn_sample/original.wav', 16000)
    distorted = load_wav('../audios/cn_sample/ry_adv.wav', 16000)
    original = original[: , :distorted.size(1)]
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

