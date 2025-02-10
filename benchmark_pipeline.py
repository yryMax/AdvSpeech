from torch.utils.data import DataLoader
from dataset.transformed_audio_dataset import TransformedAudioDataset
from Metrics.FidelityMetrics import SNRMetric, PESQMetric, SECSMetric, mos_runner
from dataset.base_audio_dataset import AudioDataset
from Metrics.EffectivenessMetric import wespeaker_runner
from synthesis.synthesizer import *
from adv_runners import *
class BenchmarkPipeline:
    def __init__(self, dataset: TransformedAudioDataset, *synthesizers: Synthesizer) -> None:
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=None, shuffle=False)
        self.name = dataset.name
        self.sample_rate = dataset.sample_rate
        self.synthesizers = list(synthesizers)

    def run_fidelity(self):
        snr_metric = SNRMetric()
        pesq_metric = PESQMetric()
        secs_metric = SECSMetric()

        mos = mos_runner(self.dataset.cache_path)
        for new_wave, raw_data in self.dataloader:
            snr_metric.update(new_wave, raw_data['source_waveform'])

            secs_metric.update(new_wave, raw_data['source_waveform'])

            new_wave_resampled = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=pesq_metric.fs)(new_wave)
            raw_data_resampled = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=pesq_metric.fs)(raw_data['source_waveform'])
            pesq_metric.update(new_wave_resampled, raw_data_resampled)

        snr = snr_metric.compute()
        pesq = pesq_metric.compute()
        secs = secs_metric.compute()

        print(f"SNR: {snr}, PESQ: {pesq}, SECS: {secs}, MOS: {mos}")
        return snr, pesq, secs, mos

    def run_effectiveness(self):
        res = {}
        for synth in self.synthesizers:
            similarity = []
            for new_wave, raw_data in self.dataloader:
                print(raw_data['speaker'])
                syn_audio = synth.syn(new_wave)
                if syn_audio is None:
                    continue
                similarity.append(wespeaker_runner(syn_audio, raw_data['source_waveform'], self.dataset.sample_rate))
            ## remove None and calculate mean/std
            similarity = [x for x in similarity if x is not None]
            similarity = torch.tensor(similarity)
            res[synth.name] = similarity.mean(), similarity.std()

        print(res)
        return res





if __name__ == '__main__':
    def mock_transform_fn(raw_data, sample_rate):
        # 1d tensor to 1d tensor
        return raw_data['source_waveform']

    root_dir = "/mnt/d/voicedata/LibriTTS/sampled_pair"
    dataset = AudioDataset(root_dir)
    #transformed_dataset = TransformedAudioDataset(dataset, mock_transform_fn, "adv_speech")
    advspeech_speech_dataset = TransformedAudioDataset(dataset, advspeech_runner, "adv_speech")
    #antifake_speech_dataset = TransformedAudioDataset(dataset, antifake_runner, "antifake")
    config = yaml.load(open("./configs/experiment_config.yaml"), Loader=yaml.FullLoader)

    openvoice = OpenVoiceSynthesizer(os.path.abspath("./external_repos/OpenVoice"), config['effectiveness'], dataset.sample_rate)
    xTTS = XTTSSynthesizer(os.path.abspath("./external_repos/TTS"), config['effectiveness'], dataset.sample_rate)
    pipeline = BenchmarkPipeline(advspeech_speech_dataset, openvoice, xTTS)

    pipeline.run_effectiveness()
    pipeline.run_fidelity()