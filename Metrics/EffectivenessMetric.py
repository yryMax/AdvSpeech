import re
import subprocess
import tempfile

import torch
import torchaudio


def wespeaker_runner(audio1: torch.Tensor, audio2: torch.Tensor, sr):
    """
    :param audio1: audio tensor 1
    :param audio2: audio tensor 2
    :return: we speaker score
    """
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=True
    ) as f1, tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f2:
        torchaudio.save(f1.name, audio1, sr, format="wav")
        torchaudio.save(f2.name, audio2, sr, format="wav")

        cmd = [
            "wespeaker",
            "--task",
            "similarity",
            "--audio_file",
            f1.name,
            "--audio_file2",
            f2.name,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            cleaned = re.sub(r"\x1b\[.*?m", "", result.stdout).strip()
            similarity_score = float(cleaned)
            return similarity_score

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] WeSpeaker execution failed: {e}")
            return None


if __name__ == "__main__":
    audio1 = torch.randn(16000)
    audio2 = torch.randn(16000)
    wespeaker_runner(audio1, audio2, 16000)
