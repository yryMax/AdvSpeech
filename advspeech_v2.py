import torch
import torchaudio

import s3tokenizer
from Psychoacoustics.psyacloss_torch import *

tokenizer = s3tokenizer.load_model("speech_tokenizer_v2_25hz").cuda()
tokenizer.train()

libri_long = s3tokenizer.load_audio("libri_long.wav").cuda()

print()


def optimize_input_representation_v2(
    x, strength=0.01, num_steps=500, lr=0.001, psy_weight=0.1, output=True
):
    loss_history = []
    original_x = x.clone().detach().cuda()

    assert libri_long.shape >= x.shape
    ref_x = libri_long[: x.shape[0]]
    mels, mels_lens = s3tokenizer.padding([s3tokenizer.log_mel_spectrogram(ref_x)])
    mel = mels.cuda()
    hidden_ref, _ = tokenizer.encoder(mel, mels_lens.to(mel.device))

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
        mel = mels.cuda()
        hidden, _ = tokenizer.encoder(mel, mels_lens.to(mel.device))

        loss = torch.nn.functional.mse_loss(hidden, hidden_ref)
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
