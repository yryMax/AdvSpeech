from enum import Enum

import torch
import torchaudio
from transformers import Wav2Vec2Model

import s3tokenizer
from audio_tokenizer import custom_processor
from audio_tokenizer import encoder_model
from Psychoacoustics.psyacloss_torch import *

tokenizer = s3tokenizer.load_model("speech_tokenizer_v2_25hz").cuda()
tokenizer.train()

libri_long = s3tokenizer.load_audio("libri_long.wav").cuda()  # 16000

print("Loading wav2vec2...")

wav2vec2 = Wav2Vec2Model.from_pretrained(
    "./audio_tokenizer_ckpt" + "/wav2vec2-large-xlsr-53"
).to("cuda")

wav2vec2.config.output_hidden_states = True

encoder_model_dict = torch.load(
    "./audio_tokenizer_ckpt" + "/semantic_tokenizer.bin",
    map_location=torch.device("cuda"),
)

encoder_model.load_state_dict(encoder_model_dict)


class TTSMode(Enum):
    COSYVOICE = "CosyVoice"
    SPARKTTS = "SparkTTS"

    @property
    def processor(self):
        if self == TTSMode.COSYVOICE:
            return lambda mel, tokenizer, mel_lens, **_: (
                tokenizer.encoder(mel, mel_lens.to("cuda"))[0]
            )
        elif self == TTSMode.SPARKTTS:
            return lambda x, wav2vec2, encoder_model, **_: (
                encoder_model(
                    (
                        (
                            (
                                feat := wav2vec2(
                                    custom_processor.process(x.unsqueeze(0)).to("cuda")
                                )
                            ).hidden_states[11]
                            + feat.hidden_states[14]
                            + feat.hidden_states[16]
                        )
                        / 3
                    ).transpose(1, 2)
                )
            )


def meldiff(x, y):
    return torch.nn.functional.mse_loss(x.squeeze(0), y.squeeze(0))


def optimize_input_representation_v2(
    x,
    strength=0.01,
    num_steps=500,
    lr=0.001,
    psy_weight=0.1,
    output=True,
    scaling=0,
    mode=TTSMode.SPARKTTS,
):
    loss_history = []
    original_x = x.clone().detach().cuda()

    assert libri_long.shape >= x.shape
    ref_x = libri_long[: x.shape[0]]
    mels, mels_lens = s3tokenizer.padding([s3tokenizer.log_mel_spectrogram(ref_x)])
    mel_ref = mels.cuda()

    token_ref = mode.processor(
        x=ref_x,
        tokenizer=tokenizer,
        mel=mel_ref,
        mel_lens=mels_lens,
        wav2vec2=wav2vec2,
        encoder_model=encoder_model,
    )

    max_amp = torch.max(torch.abs(original_x))
    eps = strength * max_amp

    w_init = torch.zeros_like(original_x).cuda()
    w = torch.nn.Parameter(w_init)

    optimizer = torch.optim.AdamW([w], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()

        x_transformed = x.cuda() + eps * torch.tanh(w)

        mels, mels_lens = s3tokenizer.padding(
            [s3tokenizer.log_mel_spectrogram(x_transformed)]
        )
        # print(meldiff(mels, mel_ref))
        mel = mels.cuda()

        token_x = mode.processor(
            x=x_transformed,
            tokenizer=tokenizer,
            mel=mel,
            mel_lens=mels_lens,
            wav2vec2=wav2vec2,
            encoder_model=encoder_model,
        )

        loss = torch.nn.functional.mse_loss(token_x, token_ref)
        if scaling != 0:
            loss += (10**scaling) * meldiff(mels, mel_ref)
        psy_loss = percloss(x_transformed.cuda(), original_x.cuda(), 16000)
        loss += psy_weight * psy_loss

        loss_history.append(loss.item())

        loss.backward(retain_graph=True)

        optimizer.step()

        if step % 5 == 0 and output:
            print(f"Step {step}, Loss: {loss.item()}")

    x_final = (original_x + eps * torch.tanh(w)).detach()
    return x_final, loss_history


def advspeechv2_runner(raw_data, sample_rate):
    audio_prompt = raw_data["source_waveform"]
    # resample to 16000
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    audio_prompt_16k = resampler(audio_prompt).squeeze(0)
    x_adv, _ = optimize_input_representation_v2(
        audio_prompt_16k, strength=0.05, num_steps=2000, psy_weight=0.1, output=False
    )
    resampler_back = torchaudio.transforms.Resample(
        orig_freq=16000, new_freq=sample_rate
    )
    return resampler_back(x_adv.cpu().unsqueeze(0))


if __name__ == "__main__":
    import torchaudio
    from util import load_wav

    audio = load_wav("./trail_ds/5694/5694_1.wav", 16000).to("cuda")
    advspeech, _ = optimize_input_representation_v2(
        audio[0], strength=0.05, num_steps=2000, psy_weight=0.1, output=False
    )
    print(advspeech.shape)
    torchaudio.save("adv.wav", advspeech.cpu().unsqueeze(0), 16000)
