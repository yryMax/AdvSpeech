# AdvSpeech


### Environment Setup
for the user:

```pycon
git submodule update --init --recursive
conda env create -f environment.yml
conda activate advspeech
pip3 install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
for dev:

`conda env update --file dev-environment.yml --prune` -> poetry update

`conda-lock lock -f dev-environment.yml` -> lock the environment


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
-  Wenet: https://github.com/wenet-e2e/wenet

then run `run benchmark_pipeline.py`

### HOW TO CONTRIBUTE TO BENCHMARK

to add a new matrix -> contribute under `Metrics`

to add a new adv method -> add a new method under `adv_runner`

to add a new synthesizer -> add a new class under `synthesizer` that inherits from `Synthesizer`

to add a new experiment -> add a new method under `BenchmarkPipeline`
