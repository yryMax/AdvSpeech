import pyworld as pw
import yaml
from acoustics import CheapTrick
from loss import SSIMLossLayer, SpectralFlatnessLoss
from util import *
import argparse


def get_envelope(audio, sr=22050):
    env_layer = CheapTrick(fs=sr)
    f0, time_stamp = pw.dio(audio.flatten().numpy().astype(np.float64), sr, frame_period=5.0)
    envelope = env_layer(torch.tensor(audio.flatten()), torch.tensor(f0).double(), torch.tensor(time_stamp).double())
    return envelope


def optimize_input(ssim_layer, audio, strength_weight=0.01, num_steps=10000, lr=0.001, device='cuda',
        debug=True, sr=22050):

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
        #noise = x - original_x
        #noise_normalized = (noise - noise.mean()) / (noise.std(unbiased=False) + 1e-8)
        # loss = MMDLoss(noise_normalized, z)

        loss = ssim_loss + strength_weight * sfm_loss_val

        if not torch.isfinite(loss):
            print(f"Optimization stopped at step {step} due to invalid loss (NaN or Inf).")
            break

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

        if step % 5 == 0 and debug:
            print(f"Step {step}, Loss: {loss.item()}")



    return x, loss_history


def process_audio(audio_path, ref_path, output_path, sr):

    audio_prompt = load_wav(audio_path, sr)
    reference = load_wav(ref_path, sr)

    promp_envelope = get_envelope(audio_prompt, sr)
    ref_envelope = get_envelope(reference, sr)

    assert ref_envelope.shape[1] >= promp_envelope.shape[1], "Reference audio should be equal or longer than prompt audio"
    ref_envelope = ref_envelope[:, :promp_envelope.shape[1]]

    normalized_ref = tensor_normalize(ref_envelope)
    ssim_layer = SSIMLossLayer(normalized_ref.double().to('cuda')).double()

    x_adv, loss_history = optimize_input(ssim_layer, audio_prompt, device='cuda', sr=sr)

    torchaudio.save(output_path, x_adv.cpu().float().detach().unsqueeze(0), sr)

    return loss_history


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process audio using SSIM optimization.")
    parser.add_argument("--config", help="Path to the configuration YAML file.")
    parser.add_argument("--input", help="Path to the input audio file.")
    parser.add_argument("--reference", help="Path to the reference audio file.")
    parser.add_argument("--output", default="output.wav", help="Path to save the processed audio file.")
    parser.add_argument("--sr", type=int, default=22050,
                        help="Sampling rate for processing (default: 22050). Input and reference must have SR >= this value.")

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        input_path = config['prompt']['audio_path']
        ref_path = config['prompt']['reference_path']
        output_path = config.get('prompt', {}).get('adv_path', "output.wav")
        sr = config['prompt'].get('sample_rate', 22050)
    else:
        if not args.input or not args.reference:
            parser.error("Either --config or both --input and --reference must be provided.")
        input_path = args.input
        ref_path = args.reference
        output_path = args.output
        sr = args.sr

    process_audio(input_path, ref_path, output_path, sr)


