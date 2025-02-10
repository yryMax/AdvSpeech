import os
from torch.utils.data import Dataset
from util import *

class AudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=22050, device='cuda'):
        self.sample_rate = sample_rate
        self.device = device
        self.data = []
        self.name = root_dir.split('/')[-1]

        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            source_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.wav') and
                            len(f.split('_')) == 2 and f.split('_')[-1][-5].isdigit()]
            for source_file in source_files:
                prefix = os.path.splitext(os.path.basename(source_file))[0]
                ref_files = [
                    os.path.join(subdir_path, f)
                    for f in os.listdir(subdir_path)
                    if f.endswith('.wav') and len(f.split('_')) == 3 and f.split('_')[-1][-5].isalpha()
                       and f.startswith(prefix)
                ]

                if ref_files:
                    self.data.append({
                        'source_file': source_file,
                        'ref_files': ref_files,
                        'text': os.path.join(subdir_path, f"{prefix}.normalized.txt")
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        audio_prompt = load_wav(item['source_file'], self.sample_rate)
        references = [load_wav(ref, self.sample_rate) for ref in item['ref_files']]
        speaker = item['source_file'].split('/')[-1].replace('.wav', '')
        return {
            'source_waveform': audio_prompt, # tensor
            'ref_waveforms': references, # list of tensors
            'text': item['text'], # string
            'speaker': speaker, # string
        }

if __name__ == '__main__':
    root_dir = "/mnt/d/voicedata/LibriTTS/sampled_pair"
    dataset = AudioDataset(root_dir)
    print(dataset[0])