from torch.utils.data import Dataset
from base_audio_dataset import AudioDataset
from adv_runners import advspeech_runner

class TransformedAudioDataset(Dataset):
    """
    for my method or competitor's method
    """

    def __init__(self, base_dataset: AudioDataset, transform_fn, use_cache=True):
        super().__init__()
        self.base_dataset = base_dataset

        # Tensor -> Tensor
        # if compare with a new method, transform_fn should be ecapsulated in adv_runner.py

        self.transform_fn = transform_fn

        self.use_cache = use_cache

        if use_cache:
            self.cache = dict()

    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):

        raw_data = self.base_dataset[idx]

        if self.use_cache and idx in self.cache:
            return self.cache[idx]

        new_wave = self.transform_fn(raw_data, sample_rate = self.base_dataset.sample_rate)

        if self.use_cache:
            self.cache[idx] = new_wave

        return new_wave, raw_data


if __name__ == '__main__':
    root_dir = "/mnt/d/voicedata/LibriTTS/sampled_pair"
    dataset = AudioDataset(root_dir)
    transformed_dataset = TransformedAudioDataset(dataset, advspeech_runner)
    print(transformed_dataset[0])