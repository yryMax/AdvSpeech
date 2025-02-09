from abc import ABC, abstractmethod
import os
import sys
import io
import time
import tempfile
import threading
import subprocess
import torchaudio
import torch
import yaml
import shutil
class Synthesizer(ABC):

    def __init__(self, model_path: str, config: dict, sr):
        self.path = model_path
        self.config = config
        self.sr = sr

    @abstractmethod
    def syn(self, ref_audio: torch.Tensor, text: str) -> torch.Tensor:
        """
        :param ref_audio: reference audio tensor
        :param text: 转录文本
        :return: TTS
        """
        pass


class XTTSSynthesizer(Synthesizer):
    def syn(self, ref_audio: torch.Tensor) -> torch.Tensor:
        text = self.config['text']

        path = self.path

        pipe_in = tempfile.mktemp(prefix="xtts_in_", suffix=".wav", dir="/tmp")
        pipe_out = tempfile.mktemp(prefix="xtts_out_", suffix=".wav", dir="/tmp")

        output_data_list = []
        reader_should_stop = threading.Event()

        def writer():
            torchaudio.save(pipe_in, ref_audio, self.sr, format="wav")
        def reader():
            max_wait_time = 3000
            waited = 0
            while not os.path.exists(pipe_out):
                if reader_should_stop.is_set():
                    print("Reader thread exiting early due to failure in subprocess.")
                    return
                time.sleep(0.5)
                waited += 0.5
                if waited >= max_wait_time:
                    print(f"Error: pipe_out file '{pipe_out}' not found after {max_wait_time} seconds!")
                    return
            try:
                with open(pipe_out, 'rb') as f:
                    output_data_list.append(f.read())
            except Exception as e:
                print(f"Error reading pipe_out file: {e}")

        t1 = threading.Thread(target=writer, daemon=True)
        t2 = threading.Thread(target=reader, daemon=True)
        t1.start()
        t2.start()

        try:
            res = subprocess.run(
                [
                    "conda", "run", "-n", "TTS",
                    "tts", "--model_name", "tts_models/multilingual/multi-dataset/xtts_v2",
                    "--text", text,
                    "--speaker_wav", pipe_in,
                    "--language_idx", "en",
                    "--use_cuda", "true",
                    "--out_path", pipe_out
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                cwd=path
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: Process failed with exit code {e.returncode}.")
            print("Child stdout =", e.stdout)
            print("Child stderr =", e.stderr)
            reader_should_stop.set()
        finally:
            t1.join()
            t2.join()

            if os.path.exists(pipe_in):
                os.remove(pipe_in)
            if os.path.exists(pipe_out):
                os.remove(pipe_out)

        out_bytes = output_data_list[0] if output_data_list else b""
        buf_out = io.BytesIO(out_bytes)
        processed_waveform, _ = torchaudio.load(buf_out)
        return processed_waveform


class OpenVoiceSynthesizer(Synthesizer):
    def syn(self, ref_audio: torch.Tensor) -> torch.Tensor:
        text = self.config['text']
        path = self.path

        pipe_in = tempfile.mktemp(prefix="openvoice_in_", suffix=".wav", dir="/tmp")
        pipe_out_dir = tempfile.mkdtemp(prefix="openvoice_out_", dir="/tmp")

        output_data_list = []
        reader_should_stop = threading.Event()

        def writer():
            torchaudio.save(pipe_in, ref_audio, self.sr, format="wav")

        def reader():
            max_wait_time = 3000
            waited = 0
            output_file = os.path.join(pipe_out_dir, "output.wav")

            while not os.path.exists(output_file):
                if reader_should_stop.is_set():
                    return
                time.sleep(0.5)
                waited += 0.5
                if waited >= max_wait_time:
                    return

            try:
                with open(output_file, "rb") as f:
                    output_data_list.append(f.read())
            except Exception as e:
                print(f"Error reading pipe_out file: {e}")

        t_writer = threading.Thread(target=writer, daemon=True)
        t_reader = threading.Thread(target=reader, daemon=True)
        t_writer.start()
        t_reader.start()
        exception_exit = False
        try:
            process = subprocess.run(
                [
                    "python", "-m", "openvoice_worker",
                    "--ref_audio", pipe_in,
                    "--text", text,
                    "--output_dir", pipe_out_dir
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                cwd=path
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: Process failed with exit code {e.returncode}.")
            print("Child stdout =", e.stdout)
            print("Child stderr =", e.stderr)
            reader_should_stop.set()
            exception_exit = True
        finally:
            t_writer.join()
            t_reader.join()

            if os.path.exists(pipe_in):
                os.remove(pipe_in)
            if os.path.exists(pipe_out_dir):
                shutil.rmtree(pipe_out_dir)

        if not output_data_list or exception_exit:
            return None

        buf_out = io.BytesIO(output_data_list[0])
        processed_waveform, sr = torchaudio.load(buf_out)
        return processed_waveform

if __name__ == '__main__':
    refaudio, sr = torchaudio.load("../adv_speech/2086_1.wav")
    config = yaml.load(open("../configs/experiment_config.yaml"), Loader=yaml.FullLoader)
    path = os.path.abspath("../external_repos/OpenVoice")
    synthesizer = OpenVoiceSynthesizer(path, config['effectiveness'], sr)
    output = synthesizer.syn(refaudio)
    # save the output
    torchaudio.save("output.wav", output, sr, format="wav")