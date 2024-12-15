import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio


cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')


reftext = "欲问后期何日是, 寄书应见雁南征"
targettext = "今天阳光明媚，我去公园散步，看见很多人放风筝。"
prompt_prefix = 'adv_example/'
output_prefix = 'cosyvoice_gen/'


def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

def syn(prompt_name):
    prompt_wav = prompt_prefix + prompt_name + '.wav'
    prompt_sr = 16000
    prompt_speech_16k = load_wav(prompt_wav, prompt_sr)

    output = cosyvoice.inference_zero_shot(targettext, reftext, prompt_speech_16k, stream=False, speed=1)
    for j, k in enumerate(output):
        torchaudio.save(output_prefix + "/" +  prompt_name + '_cosyvoice.wav'.format(j), k['tts_speech'], 22050)

if __name__ == '__main__':
    syn('original')