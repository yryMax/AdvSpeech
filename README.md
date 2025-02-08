# AdvSpeech


### Environment Setup
for the user:

` conda env create -f environment.yml` -> poetry install

for dev:

`conda env update --file environment.yml --prune` -> poetry update

`conda-lock lock -f environment.yml` -> lock the environment

### HOW TO USE
provide the info directly

`python main.py --input audios/en_sample/libri_5694.wav --reference audios/en_sample/ref_ws.wav`

or use the config

`python main.py --config audios/en_sample/config.yaml`

sample audio is hosted on  https://yrymax.github.io/AdvSpeech/sample_web/
dataset: https://huggingface.co/datasets/Renyi444/AdvSpeech




