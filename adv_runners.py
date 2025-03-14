import io
import os
import subprocess
import sys
import tempfile
import threading
import time

from torch import Tensor

from loss import SSIMLossLayer
from main import get_envelope
from main import optimize_input
from util import *


def advspeech_runner(raw_data, sample_rate):
    audio_prompt = raw_data["source_waveform"]
    # for now only use the first one
    reference = raw_data["ref_waveforms"][0]
    promp_envelope = get_envelope(audio_prompt, sample_rate)
    ref_envelope = get_envelope(reference, sample_rate)
    assert (
        ref_envelope.shape[1] >= promp_envelope.shape[1]
    ), "Reference audio should be equal or longer than prompt audio"
    ref_envelope = ref_envelope[:, : promp_envelope.shape[1]]
    normalized_ref = tensor_normalize(ref_envelope)
    ssim_layer = SSIMLossLayer(normalized_ref.double().to("cuda")).double()
    x_adv, loss_history = optimize_input(
        ssim_layer, audio_prompt, device="cuda", sr=sample_rate
    )
    assert audio_prompt.size() == x_adv.size()
    return x_adv.float().cpu().unsqueeze(0)


def antifake_runner(base_data_dict, sample_rate):
    raw_data = base_data_dict["source_waveform"]
    if raw_data.dim() == 1:
        raw_data = raw_data.unsqueeze(0)
    if raw_data.size(0) > 1:
        raw_data = raw_data.mean(dim=0, keepdim=True)

    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    raw_data_16k = resampler(raw_data)

    pipe_in = tempfile.mktemp(prefix="antifake_in_", suffix=".wav", dir="/tmp")
    pipe_out = tempfile.mktemp(prefix="antifake_out_", suffix=".wav", dir="/tmp")

    output_data_list = []
    reader_should_stop = threading.Event()

    def writer():
        torchaudio.save(pipe_in, raw_data_16k, 16000, format="wav")

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

    try:
        res = subprocess.run(
            ["conda", "run", "-n", "cosyvoice", "python", "run.py", pipe_in, pipe_out],
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            check=True,
            cwd="external_repos/antifake",
        )
        print("returncode =", res.returncode)

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
    processed_waveform, sr = torchaudio.load(buf_out)
    # resample to original sample rate
    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=sample_rate)
    processed_waveform = resampler(processed_waveform)
    return processed_waveform


def pop_runner(base_data_dict, sample_rate):
    pipe_in = tempfile.mktemp(prefix="pop_in_", suffix=".txt", dir="/tmp")
    pipe_out = tempfile.mktemp(prefix="pop_out_", suffix=".wav", dir="/tmp")

    output_data_list = []
    reader_should_stop = threading.Event()
    path = "../../" + base_data_dict["path"][2:]
    filelist_str = path + "|0|" + base_data_dict["text"]

    def writer():
        # write the string to the pipe_in txt file
        with open(pipe_in, "w") as f:
            f.write(filelist_str)

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
    project_path = "external_repos/Pivotal_Objective_Perturbation"
    exception = False
    try:
        res = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                "cosyvoice2",
                "python",
                "protect_one.py",
                "--filelist",
                pipe_in,
                "--output_path",
                pipe_out,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            cwd=project_path,
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
    processed_waveform, _ = torchaudio.load(buf_out)
    return processed_waveform


def safespecch_runner(base_data_dict, sample_rate):
    pipe_in = tempfile.mktemp(prefix="safespeech_in_", suffix=".txt", dir="/tmp")
    pipe_out = tempfile.mktemp(prefix="safespeech_out_", suffix=".wav", dir="/tmp")

    output_data_list = []
    reader_should_stop = threading.Event()
    path = "../../" + base_data_dict["path"][2:]
    filelist_str = path + "|0|" + base_data_dict["text"]
    print(filelist_str)

    def writer():
        # write the string to the pipe_in txt file
        with open(pipe_in, "w") as f:
            f.write(filelist_str)

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
    project_path = "external_repos/SafeSpeech"
    exception = False
    try:
        res = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                "safespeech",
                "python",
                "protect_one.py",
                "--filepath",
                pipe_in,
                "--output_path",
                pipe_out,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            cwd=project_path,
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
    processed_waveform, sr = torchaudio.load(buf_out)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
    torchaudio.save("qwq.wav", processed_waveform, sample_rate, format="wav")
    return resampler(processed_waveform)


if __name__ == "__main__":
    audio = load_wav("audios/en_sample/libri_5694.wav", 16000)
    output = antifake_runner(audio, 16000)
    torchaudio.save("antifake_output.wav", output, 16000, format="wav")
