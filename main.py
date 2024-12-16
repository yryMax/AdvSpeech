import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from librosa.filters import mel as librosa_mel_fn
import matplotlib.pyplot as plt
import IPython.display as ipd
import math
import numpy as np
import pyworld as pw
import yaml
from acoustics import CheapTrick
from loss import SSIMLossLayer, SpectralFlatnessLoss
from util import *



def get_envelope(audio, sr=22050):
    env_layer = CheapTrick(fs=sr)
    f0, time_stamp = pw.dio(audio.flatten().numpy().astype(np.float64), sr, frame_period=5.0)
    envelope = env_layer(torch.tensor(audio.flatten()), torch.tensor(f0).double(), torch.tensor(time_stamp).double())
    return envelope


def optimize_input(ssim_layer, audio, strength_weight=0.01, num_steps=10000, lr=0.001, device='cuda',
        output=True, sr=22050):

    f0, time_stamp = pw.dio(audio.flatten().numpy().astype(np.float64), sr, frame_period=5.0)
    f0 = torch.tensor(f0).double().to(device)
    time_stamp = torch.tensor(time_stamp).double().to(device)

    envelope_layer = CheapTrick(fs=sr).double().to(device)
    ssim_layer = ssim_layer.double().to(device)

    x = audio.clone().detach().flatten().double()
    x = x.to(device)
    x.requires_grad_(True)

    optimizer = torch.optim.Adam([x], lr=lr)

    loss_history = []
    original_x = x.clone().detach().to(device).requires_grad_(False)

    #mfcc_loss = MFCCLoss().double().to(device)
    sfm_loss = SpectralFlatnessLoss().double().to(device)
    patience = 5
    no_progress_counter = 0
    no_progress_threshold = 1e-3

    for step in range(num_steps):
        optimizer.zero_grad()

        envelop = envelope_layer(x, f0, time_stamp)
        envelop = tensor_normalize(envelop)
        ssim_loss = ssim_layer(envelop)

        #mcd_loss = mfcc_loss(original_x, x)
        sfm_loss_val = sfm_loss(original_x, x)
        noise = x - original_x
        noise_normalized = (noise - noise.mean()) / (noise.std(unbiased=False) + 1e-8)
        # loss = MMDLoss(noise_normalized, z)

        loss = ssim_loss + sfm_loss_val * strength_weight

        if len(loss_history) > 0 and loss_history[-1] - loss.item() < no_progress_threshold:
            no_progress_counter += 1
        else:
            no_progress_counter = 0

        if no_progress_counter > patience:
            print(f"Optimization stopped at step {step}, loss: {loss.item()}")
            break

        loss_history.append(loss.item())

        loss.backward()
        optimizer.step()

        if step % 5 == 0 and output:
            print(f"Step {step}, Loss: {loss.item()}")



    return x, loss_history


if __name__ == '__main__':

    # load config file
    with open('audios/en_sample/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    adv_path = config['prompt']['adv_path']
    print(adv_path)
    sr = config['prompt']['sample_rate']
    audio_prompt = load_wav(config['prompt']['audio_path'], sr)
    reference = load_wav(config['prompt']['reference_path'], sr)

    print(audio_prompt.shape, reference.shape)

    promp_envelope = get_envelope(audio_prompt, sr)
    ref_envelope = get_envelope(reference, sr)

    print(promp_envelope.shape, ref_envelope.shape)

    assert ref_envelope.shape[1] >= promp_envelope.shape[1], "Reference audio should be equal or longer than prompt audio"

    ref_envelope = ref_envelope[:, :promp_envelope.shape[1]]

    normalized_ref = tensor_normalize(ref_envelope)
    ssim_layer = SSIMLossLayer(normalized_ref.double().to('cuda')).double()

    x_adv, loss_history = optimize_input(ssim_layer, audio_prompt, device='cuda', sr=sr)

    torchaudio.save(adv_path, x_adv.cpu().float().detach().unsqueeze(0), sr)


