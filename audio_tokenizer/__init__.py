import os

import torch

from .feat_encoder import Encoder


class CustomWav2Vec2Processor:
    def __init__(self):
        self.sampling_rate = 16000
        self.do_normalize = True

    def zero_mean_unit_var_norm(self, input_values: torch.Tensor) -> torch.Tensor:
        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)  # [length] -> [1, length]

        mean = input_values.mean(dim=-1, keepdim=True)
        std = input_values.std(dim=-1, keepdim=True) + 1e-8
        normalized = (input_values - mean) / std

        return normalized

    def process(
        self,
        raw_speech: torch.Tensor,
    ) -> torch.Tensor:
        if raw_speech.dim() == 1:
            raw_speech = raw_speech.unsqueeze(0)  # [length] -> [1, length]

        if self.do_normalize:
            input_values = self.zero_mean_unit_var_norm(raw_speech)
        else:
            input_values = raw_speech

        input_values = input_values.to(dtype=torch.float32)

        return input_values


package_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(package_dir, "..", "audio_tokenizer_ckpt")
model_dir = os.path.normpath(model_dir)

print("Loading encoder...")
encoder_model = Encoder(
    input_channels=1024,
    vocos_dim=384,
    vocos_intermediate_dim=2048,
    vocos_num_layers=12,
    out_channels=1024,
    sample_ratios=[1, 1],
).to("cuda")

custom_processor = CustomWav2Vec2Processor()

__all__ = ["encoder_model", "custom_processor"]
