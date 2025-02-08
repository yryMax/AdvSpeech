from main import optimize_input, get_envelope
from util import *
from loss import SSIMLossLayer

def advspeech_runner(raw_data, sample_rate):
    audio_prompt = raw_data['source_waveform']
    reference = raw_data['ref_waveforms'][0]
    promp_envelope = get_envelope(audio_prompt, sample_rate)
    ref_envelope = get_envelope(reference, sample_rate)
    assert ref_envelope.shape[1] >= promp_envelope.shape[
        1], "Reference audio should be equal or longer than prompt audio"
    ref_envelope = ref_envelope[:, :promp_envelope.shape[1]]
    normalized_ref = tensor_normalize(ref_envelope)
    ssim_layer = SSIMLossLayer(normalized_ref.double().to('cuda')).double()
    x_adv, loss_history = optimize_input(ssim_layer, audio_prompt, device='cuda', sr=sample_rate)
    return x_adv

def antifake_runner(raw_data, sample_rate):
    pass