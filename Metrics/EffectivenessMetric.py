import re
import subprocess
import tempfile

import torch
import torchaudio
import wespeaker

model = wespeaker.load_model("english")
model.set_device("cuda:0")


def wespeaker_runner(audio1: torch.Tensor, audio2: torch.Tensor, sr):
    """
    :param audio1: audio tensor 1
    :param audio2: audio tensor 2
    :return: wespeaker score
    """
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=True
    ) as f1, tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f2:
        torchaudio.save(f1.name, audio1, sr, format="wav")
        torchaudio.save(f2.name, audio2, sr, format="wav")

        return model.compute_similarity(f1.name, f2.name)


if __name__ == "__main__":
    audio1 = torch.randn(16000)
    audio2 = torch.randn(16000)
    wespeaker_runner(audio1, audio2, 16000)
