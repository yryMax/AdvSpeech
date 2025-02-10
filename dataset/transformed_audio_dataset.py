from torch.utils.data import Dataset
from dataset.base_audio_dataset import AudioDataset
from adv_runners import advspeech_runner
import os
from util import load_wav
import torchaudio


class TransformedAudioDataset(Dataset):
    """
    for my method or competitor's method
    """

    def __init__(self, base_dataset: AudioDataset, transform_fn, name, use_cache=True):
        super().__init__()
        self.base_dataset = base_dataset

        # Tensor -> Tensor
        # if compare with a new method, transform_fn should be ecapsulated in adv_runner.py

        self.transform_fn = transform_fn

        self.use_cache = use_cache
        self.name = name
        self.cache_path = os.path.join(os.path.abspath(os.getcwd()), name)
        self.sample_rate = base_dataset.sample_rate
        if use_cache:
            self.cache = dict()
            # create the directory for the cache
            os.makedirs(name, exist_ok=True)

    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):

        raw_data = self.base_dataset[idx]

        if self.use_cache and idx in self.cache:
            return self.cache[idx], raw_data

        # if the file is contained under directory
        if os.path.exists(self.name + "/" + raw_data['speaker'] + ".wav"):
            new_wave = load_wav(self.name + "/" + raw_data['speaker'] + ".wav", self.base_dataset.sample_rate)
        else:
            new_wave = self.transform_fn(raw_data, sample_rate = self.base_dataset.sample_rate)

        assert new_wave.shape[0] == 1

        new_wave = new_wave[:, :raw_data['source_waveform'].shape[-1]]
        raw_data['source_waveform'] = raw_data['source_waveform'][:, :new_wave.shape[-1]]

        if self.use_cache:
            self.cache[idx] = new_wave
            # save the new wave
            torchaudio.save(self.name + "/" + raw_data['speaker'] + ".wav", new_wave, self.base_dataset.sample_rate)
        return new_wave, raw_data


if __name__ == '__main__':

    def mock_transform_fn(raw_data, sample_rate):
        # 1d tensor to 1d tensor
        return raw_data['source_waveform']

    root_dir = "/mnt/d/voicedata/LibriTTS/sampled_pair"
    dataset = AudioDataset(root_dir)
    transformed_dataset = TransformedAudioDataset(dataset, advspeech_runner, "adv_speech")
    print(transformed_dataset[0])