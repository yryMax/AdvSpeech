import torch
from feat_encoder import Encoder
from transformers import Wav2Vec2Model


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


if __name__ == "__main__":
    model_dir = "../audio_tokenizer_ckpt"
    print("Loading encoder...")
    encoder_model = Encoder(
        input_channels=1024,
        vocos_dim=384,
        vocos_intermediate_dim=2048,
        vocos_num_layers=12,
        out_channels=1024,
        sample_ratios=[1, 1],
    ).to("cuda")

    print("Loading wav2vec2...")
    wav2vec2 = Wav2Vec2Model.from_pretrained(model_dir + "/wav2vec2-large-xlsr-53").to(
        "cuda"
    )

    wav2vec2.config.output_hidden_states = True

    encoder_model_dict = torch.load(
        model_dir + "/semantic_tokenizer.bin", map_location=torch.device("cuda")
    )

    encoder_model.load_state_dict(encoder_model_dict)

    custom_processor = CustomWav2Vec2Processor()

    import torch
    import torchaudio

    def load_wav(wav, target_sr):
        speech, sample_rate = torchaudio.load(wav)
        speech = speech.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            assert (
                sample_rate >= target_sr
            ), "wav sample rate {} must be greater than {}".format(
                sample_rate, target_sr
            )
            speech = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sr
            )(speech)
        return speech

    # test

    wav = load_wav("../qwq.wav", 16000)
    wav_tensor = custom_processor.process(wav)

    feat = wav2vec2(wav_tensor.to("cuda"))
    feats_mix = (
        feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
    ) / 3

    z = encoder_model(feats_mix.transpose(1, 2))

    print(z)
