from torch.utils.data import DataLoader
from tqdm import tqdm

from adv_runners import *
from advspeech_v2 import advspeechv2_runner
from dataset.base_audio_dataset import AudioDataset
from dataset.transformed_audio_dataset import TransformedAudioDataset
from Metrics.EffectivenessMetric import wer_runner
from Metrics.EffectivenessMetric import wespeaker_runner
from Metrics.FidelityMetrics import mos_runner
from Metrics.FidelityMetrics import PESQMetric
from Metrics.FidelityMetrics import SECSMetric
from Metrics.FidelityMetrics import SNRMetric
from synthesis.synthesizer import *


class BenchmarkPipeline:
    def __init__(
        self, dataset: TransformedAudioDataset, *synthesizers: Synthesizer
    ) -> None:
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=None, shuffle=False)
        self.name = dataset.name
        self.sample_rate = dataset.sample_rate
        self.synthesizers = list(synthesizers)

    def run_fidelity(self):
        snr_metric = SNRMetric()
        pesq_metric = PESQMetric()
        secs_metric = SECSMetric()
        print("Running fidelity metrics")

        for new_wave, raw_data in tqdm(self.dataloader, desc="Loading Data"):
            snr_metric.update(new_wave, raw_data["source_waveform"])

            secs_metric.update(new_wave, raw_data["source_waveform"])

            new_wave_resampled = torchaudio.transforms.Resample(
                orig_freq=self.sample_rate, new_freq=pesq_metric.fs
            )(new_wave)
            raw_data_resampled = torchaudio.transforms.Resample(
                orig_freq=self.sample_rate, new_freq=pesq_metric.fs
            )(raw_data["source_waveform"])
            pesq_metric.update(new_wave_resampled, raw_data_resampled)

        snr = snr_metric.compute()
        pesq = pesq_metric.compute()
        secs = secs_metric.compute()

        print(f"SNR: {snr}, PESQ: {pesq}, SECS: {secs}")
        return snr, pesq, secs

    def run_effectiveness(self, preserve_audio=True):
        res = {}
        print("Running effectiveness metrics")
        for synth in self.synthesizers:
            if preserve_audio:
                # make dir
                os.makedirs("./" + self.dataset.name + "/" + synth.name, exist_ok=True)
            print(f"Running {synth.name}")
            # folder = "./" + self.dataset.name + "/" + synth.name

            # similarity = []
            wer = []
            wil = []
            cer = []
            bleu = []
            for new_wave, raw_data in tqdm(self.dataloader, desc="Loading Data"):
                syn_audio = synth.syn(new_wave, raw_data["text"])
                if syn_audio is None:
                    print("Synthesis failed" + synth.name + " " + raw_data["speaker"])
                    continue
                if preserve_audio:
                    torchaudio.save(
                        "./"
                        + self.dataset.name
                        + "/"
                        + synth.name
                        + "/"
                        + raw_data["speaker"]
                        + ".wav",
                        syn_audio,
                        self.dataset.sample_rate,
                    )
                # similarity.append(
                #    wespeaker_runner(
                #        syn_audio, raw_data["source_waveform"], self.dataset.sample_rate
                #    )
                # )
                correctness = wer_runner(syn_audio, synth.config["text"], synth.sr)
                wer.append(correctness["wer"])
                wil.append(correctness["wil"])
                cer.append(correctness["cer"])
                bleu.append(correctness["bleu"])
                # wer.append(wer_runner(syn_audio, synth.config["text"], synth.sr))
                # moss = mos_runner(os.path.abspath(folder))
            ## remove None and calculate mean/std
            # similarity = [x for x in similarity if x is not None]
            # similarity = torch.tensor(similarity)
            wer = torch.tensor(wer)
            wil = torch.tensor(wil)
            cer = torch.tensor(cer)
            bleu = torch.tensor(bleu)
            # moss = torch.tensor(moss)

            res[synth.name] = {
                "wer: ": (wer.mean(), wer.std()),
                "wil: ": (wil.mean(), wil.std()),
                "cer: ": (cer.mean(), cer.std()),
                "bleu: ": (bleu.mean(), bleu.std()),
            }

        print(res)
        return res


if __name__ == "__main__":

    def mock_transform_fn(raw_data, sample_rate):
        # 1d tensor to 1d tensor
        return raw_data["source_waveform"]

    root_dir = "./sampled_pair"
    dataset = AudioDataset(root_dir, sample_rate=16000)
    transformed_dataset = TransformedAudioDataset(
        dataset, mock_transform_fn, "spark_advspeechv2"
    )
    # advspeech = TransformedAudioDataset(dataset, advspeech_runner, "adv_speech")
    # antifake_speech_dataset = TransformedAudioDataset(
    #    dataset, antifake_runner, "antifake"
    # )
    # safespeech = TransformedAudioDataset(dataset, safespecch_runner, "safespeech")
    # advspeech_v2 = TransformedAudioDataset(
    #    dataset, advspeechv2_runner, "adv_speech_spark08"
    # )
    config = yaml.load(open("./configs/experiment_config.yaml"), Loader=yaml.FullLoader)
    """
    cosyvoice = CosyVoiceSynthesizer(
        os.path.abspath("./external_repos/CosyVoice"),
        config["effectiveness"],
        dataset.sample_rate,
    )

    openvoice = OpenVoiceSynthesizer(
        os.path.abspath("./external_repos/OpenVoice"),
        config["effectiveness"],
        dataset.sample_rate,
    )

    xTTS = XTTSSynthesizer(
        os.path.abspath("./external_repos/TTS"),
        config["effectiveness"],
        dataset.sample_rate,
    )
    """

    sparktts = SparkTTSSynthesizer(
        os.path.abspath("./external_repos/Spark-TTS"),
        config["effectiveness"],
        dataset.sample_rate,
    )

    pipeline = BenchmarkPipeline(transformed_dataset, sparktts)
    pipeline.run_effectiveness()
    pipeline.run_fidelity()
