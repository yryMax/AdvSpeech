import logging
import os

import torch
import torchaudio
import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from loss import SSIMLossLayer
from main import get_envelope
from main import optimize_input
from Metrics.FidelityMetrics import PESQMetric
from Metrics.FidelityMetrics import SECSMetric
from Metrics.FidelityMetrics import SNRMetric
from util import *


class AudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, device="cuda"):
        self.sample_rate = sample_rate
        self.device = device
        self.data = []
        self.output_suffix = "advspeech_ssim_only"

        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            source_files = [
                os.path.join(subdir_path, f)
                for f in os.listdir(subdir_path)
                if f.endswith(".wav")
                and len(f.split("_")) == 2
                and f.split("_")[-1][-5].isdigit()
            ]
            for source_file in source_files:
                prefix = os.path.splitext(os.path.basename(source_file))[0]
                ref_files = [
                    os.path.join(subdir_path, f)
                    for f in os.listdir(subdir_path)
                    if f.endswith(".wav")
                    and len(f.split("_")) == 3
                    and f.split("_")[-1][-5].isalpha()
                    and f.startswith(prefix)
                ]

                if ref_files:
                    self.data.append(
                        {
                            "source_file": source_file,
                            "ref_files": ref_files,
                            "adv_path": os.path.join(
                                subdir_path, f"{prefix}_{self.output_suffix}.wav"
                            ),
                        }
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        audio_prompt = load_wav(item["source_file"], self.sample_rate)
        references = [load_wav(ref, self.sample_rate) for ref in item["ref_files"]]

        return {
            "source_waveform": audio_prompt,
            "ref_waveforms": references,
            "adv_path": item["adv_path"],
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("output_log_with_metrics.txt"),  # 保存日志到文件
            logging.StreamHandler(),  # 同时输出到控制台
        ],
    )

    root_dir = "./sampled_pair"
    dataset = AudioDataset(root_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    snr_metric = SNRMetric()
    pesq_metric = PESQMetric()
    secs_metric = SECSMetric()

    for i, item in enumerate(tqdm(dataloader, desc="Processing Data")):
        source_waveform = item["source_waveform"].squeeze(0)
        ref_waveforms = [ref for ref in item["ref_waveforms"]]
        adv_path = item["adv_path"][0]

        promp_envelope = get_envelope(source_waveform, dataset.sample_rate)

        index = 0
        for ref_waveform in ref_waveforms:
            adv_path_with_index = adv_path.replace(".wav", f"_{index}.wav")
            if os.path.exists(adv_path_with_index):
                continue
            ref_envelope = get_envelope(ref_waveform, dataset.sample_rate)

            assert (
                ref_envelope.shape[1] >= promp_envelope.shape[1]
            ), "Reference audio should be equal or longer than prompt audio"

            ref_envelope = ref_envelope[:, : promp_envelope.shape[1]]

            normalized_ref = tensor_normalize(ref_envelope)

            ssim_layer = SSIMLossLayer(
                normalized_ref.double().to(dataset.device)
            ).double()

            x_adv, loss_history = optimize_input(
                ssim_layer,
                source_waveform,
                device=dataset.device,
                sr=dataset.sample_rate,
            )
            if len(loss_history) == 0:
                continue

            logging.info(f"Saving {adv_path_with_index}" + "...")
            logging.info(f"final loss: {loss_history[-1]}")
            torchaudio.save(
                adv_path_with_index,
                x_adv.cpu().float().detach().unsqueeze(0),
                dataset.sample_rate,
            )
            snr_metric.update(
                x_adv.cpu().float().detach().unsqueeze(0), ref_waveform.squeeze(0)
            )
            pesq_metric.update(
                x_adv.cpu().float().detach().unsqueeze(0), ref_waveform.squeeze(0)
            )
            secs_metric.update(
                x_adv.cpu().float().detach().unsqueeze(0), ref_waveform.squeeze(0)
            )

            index += 1

    snr_value = snr_metric.compute().item()
    pesq_value = pesq_metric.compute().item()
    secs_value = secs_metric.compute().item()

    logging.info(
        f"Final Metrics - SNR: {snr_value}, PESQ: {pesq_value}, SECS: {secs_value}"
    )
