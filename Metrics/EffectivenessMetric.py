import re
import subprocess
import tempfile

import torch
import torchaudio
import wespeaker
from jiwer import *

import wenet

wer_standardize_contiguous = Compose(
    [
        ToLowerCase(),
        ExpandCommonEnglishContractions(),
        RemoveKaldiNonWords(),
        RemoveWhiteSpace(replace_by_space=True),
        RemoveMultipleSpaces(),
        RemovePunctuation(),
        Strip(),
        ReduceToSingleSentence(),
        ReduceToListOfListOfWords(),
    ]
)

model_wespeaker = wespeaker.load_model("english")
model_wespeaker.set_device("cuda:0")
model_asr = wenet.load_model("english")


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

        return model_wespeaker.compute_similarity(f1.name, f2.name)


def WER_runner(audio: torch.Tensor, text_target: str, sr):
    """
    Args:
        audio1: audio tensor
        text_target: text that corresponds to the audio

    Returns: word error rate
    """

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        torchaudio.save(f.name, audio, sr, format="wav")

        text = model_asr.transcribe(f.name)["text"].replace("▁", " ")
        return wer(
            text_target,
            text,
            truth_transform=wer_standardize_contiguous,
            hypothesis_transform=wer_standardize_contiguous,
        )


if __name__ == "__main__":
    audio1, sr = torchaudio.load("../adv_speech/6319_2.wav")
    print(
        WER_runner(
            audio1,
            "My young plants require heat, or they would not live; and the pots we are kept in protect us from those cruel wire worms who delight to destroy our roots.",
            sr,
        )
    )
    # audio2 = torch.randn(16000)
    # wespeaker_runner(audio1, audio2, 16000)
