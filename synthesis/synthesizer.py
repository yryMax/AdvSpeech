import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC
from abc import abstractmethod

import torch
import torchaudio
import yaml

from util import *


class Synthesizer(ABC):
    def __init__(self, model_path: str, config: dict, sr, name):
        self.path = model_path
        self.config = config
        self.name = name
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
    def __init__(self, model_path: str, config: dict, sr):
        super(XTTSSynthesizer, self).__init__(model_path, config, sr, "XTTS")

    def syn(self, ref_audio: torch.Tensor, text: str) -> torch.Tensor:
        path = self.path

        pipe_in = tempfile.mktemp(prefix="xtts_in_", suffix=".wav", dir="/tmp")
        pipe_out = tempfile.mktemp(prefix="xtts_out_", suffix=".wav", dir="/tmp")

        output_data_list = []
        reader_should_stop = threading.Event()
        print(self.sr)

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
                    print(
                        f"Error: pipe_out file '{pipe_out}' not found after {max_wait_time} seconds!"
                    )
                    return
            try:
                with open(pipe_out, "rb") as f:
                    output_data_list.append(f.read())
            except Exception as e:
                print(f"Error reading pipe_out file: {e}")

        t1 = threading.Thread(target=writer, daemon=True)
        t2 = threading.Thread(target=reader, daemon=True)
        t1.start()
        t2.start()

        exception = False
        try:
            res = subprocess.run(
                [
                    "conda",
                    "run",
                    "-n",
                    "TTS",
                    "tts",
                    "--model_name",
                    "tts_models/multilingual/multi-dataset/xtts_v2",
                    "--text",
                    self.config["text"],
                    "--speaker_wav",
                    pipe_in,
                    "--language_idx",
                    "en",
                    "--use_cuda",
                    "true",
                    "--out_path",
                    pipe_out,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                cwd=path,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: Process failed with exit code {e.returncode}.")
            print("Child stdout =", e.stdout)
            print("Child stderr =", e.stderr)
            reader_should_stop.set()
            exception = True
        finally:
            t1.join()
            t2.join()

            if os.path.exists(pipe_in):
                os.remove(pipe_in)
            if os.path.exists(pipe_out):
                os.remove(pipe_out)

        if not output_data_list or exception:
            return None

        out_bytes = output_data_list[0] if output_data_list else b""
        buf_out = io.BytesIO(out_bytes)
        processed_waveform = load_wav(buf_out, self.sr)
        return processed_waveform


class OpenVoiceSynthesizer(Synthesizer):
    def __init__(self, model_path: str, config: dict, sr):
        super(OpenVoiceSynthesizer, self).__init__(model_path, config, sr, "OpenVoice")

    def syn(self, ref_audio: torch.Tensor, text: str) -> torch.Tensor:
        path = self.path

        pipe_in = tempfile.mktemp(prefix="openvoice_in_", suffix=".wav", dir="/tmp")
        pipe_out = tempfile.mktemp(prefix="openvoice_out_", suffix=".wav", dir="/tmp")

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
                    print(
                        f"Error: pipe_out file '{pipe_out}' not found after {max_wait_time} seconds!"
                    )
                    return
            try:
                with open(pipe_out, "rb") as f:
                    output_data_list.append(f.read())
            except Exception as e:
                print(f"Error reading pipe_out file: {e}")

        t_writer = threading.Thread(target=writer, daemon=True)
        t_reader = threading.Thread(target=reader, daemon=True)
        t_writer.start()
        t_reader.start()
        exception_exit = False
        try:
            subprocess.run(
                [
                    "conda",
                    "run",
                    "-n",
                    "openvoice",
                    "python",
                    "openvoice_worker.py",
                    "--ref_audio",
                    pipe_in,
                    "--text",
                    self.config["text"],
                    "--output_dir",
                    pipe_out,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                cwd=path,
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
            if os.path.exists(pipe_out):
                os.remove(pipe_out)

        if not output_data_list or exception_exit:
            return None

        buf_out = io.BytesIO(output_data_list[0])
        processed_waveform = load_wav(buf_out, self.sr)
        return processed_waveform


class CosyVoiceSynthesizer(Synthesizer):
    def __init__(self, model_path: str, config: dict, sr):
        super(CosyVoiceSynthesizer, self).__init__(model_path, config, sr, "CosyVoice")

    def syn(self, ref_audio: torch.Tensor, text: str) -> torch.Tensor:
        ref_text = self.config["text"]

        path = self.path

        pipe_in = tempfile.mktemp(prefix="cosyvoice_in_", suffix=".wav", dir="/tmp")
        pipe_out = tempfile.mktemp(prefix="cosyvoice_out_", suffix=".wav", dir="/tmp")

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
                    print(
                        f"Error: pipe_out file '{pipe_out}' not found after {max_wait_time} seconds!"
                    )
                    return
            try:
                with open(pipe_out, "rb") as f:
                    output_data_list.append(f.read())
            except Exception as e:
                print(f"Error reading pipe_out file: {e}")

        t1 = threading.Thread(target=writer, daemon=True)
        t2 = threading.Thread(target=reader, daemon=True)
        t1.start()
        t2.start()

        exception = False
        try:
            res = subprocess.run(
                [
                    "conda",
                    "run",
                    "-n",
                    "cosyvoice2",
                    "python",
                    "cosyvoice_worker.py",
                    "--ref_audio",
                    pipe_in,
                    "--text",
                    ref_text,
                    "--output_dir",
                    pipe_out,
                    "--prompt_text",
                    text,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                cwd=path,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: Process failed with exit code {e.returncode}.")
            print("Child stdout =", e.stdout)
            print("Child stderr =", e.stderr)
            reader_should_stop.set()
            exception = True
        finally:
            t1.join()
            t2.join()

            if os.path.exists(pipe_in):
                os.remove(pipe_in)
            if os.path.exists(pipe_out):
                os.remove(pipe_out)

        if not output_data_list or exception:
            return None

        out_bytes = output_data_list[0] if output_data_list else b""
        buf_out = io.BytesIO(out_bytes)
        processed_waveform = load_wav(buf_out, self.sr)
        return processed_waveform


if __name__ == "__main__":
    refaudio, sr = torchaudio.load("../adv_speech/2086_1.wav")
    config = yaml.load(
        open("../configs/experiment_config.yaml"), Loader=yaml.FullLoader
    )
    path = os.path.abspath("../external_repos/OpenVoice")
    synthesizer = OpenVoiceSynthesizer(path, config["effectiveness"], sr)
    output = synthesizer.syn(refaudio)
    torchaudio.save("output.wav", output, sr, format="wav")
