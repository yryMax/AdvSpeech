# AdvSpeech


### Environment Setup
for the user:
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

` conda env create -f environment.yml` -> poetry install

for dev:

`conda env update --file environment.yml --prune` -> poetry update

`conda-lock lock -f environment.yml` -> lock the environment

### HOW TO USE
if you only want to protect the audio, you could either provide the info directly

`python main.py --input audios/en_sample/libri_5694.wav --reference audios/en_sample/ref_ws.wav`

or use the config

`python main.py --config audios/en_sample/config.yaml`

sample audio is hosted on  https://yrymax.github.io/AdvSpeech/sample_web/
dataset: https://huggingface.co/datasets/Renyi444/AdvSpeech

if you want to see the benchmark, you need you install the env for all the external repositories.

-  antifake: https://github.com/WUSTL-CSPL/AntiFake
-  NISQA: https://github.com/gabrielmittag/NISQA
-  TTS: https://github.com/coqui-ai/TTS
-  OpenVoice: https://github.com/myshell-ai/OpenVoice
-  CosyVoice: https://github.com/FunAudioLLM/CosyVoice
-  Wespeaker: https://github.com/wenet-e2e/wespeaker






