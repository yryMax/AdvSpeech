import tempfile

import evaluate
import regex
import torch
import torchaudio
import wespeaker
import whisper
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


def norm_tok(s: str):
    s = s.lower()
    s = regex.sub(r"\p{P}+", " ", s)
    return s.strip().split()


# model_wespeaker.set_device("cuda:0")
# model_asr = wenet.load_model("english")
bleu_metric = evaluate.load("google_bleu")
model_asr = whisper.load_model("large")


def wespeaker_runner(audio1: torch.Tensor, audio2: torch.Tensor, sr):
    """
    ***: Deprecated!!!!!!!!!!!
    :param audio1: audio tensor 1
    :param audio2: audio tensor 2
    :return: wespeaker score
    """
    model_wespeaker = wespeaker.load_model("english")
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=True
    ) as f1, tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f2:
        torchaudio.save(f1.name, audio1, sr, format="wav")
        torchaudio.save(f2.name, audio2, sr, format="wav")

        return model_wespeaker.compute_similarity(f1.name, f2.name)


def wer_runner(audio: torch.Tensor, text_target: str, sr):
    """
    Args:
        audio1: audio tensor
        text_target: text that corresponds to the audio

    Returns: word error rate
    """
    min_duration = 0.15  # 150ms

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        torchaudio.save(f.name, audio, sr, format="wav")
        res = model_asr.transcribe(f.name, language="en", task="transcribe")
        text = res["text"]
        print(text)
        # print("REF:", wer_standardize_contiguous(text_target))
        # print("HYP:", wer_standardize_contiguous(text))
        sentence_bleu_score = bleu_metric.compute(
            predictions=[text],
            references=[[text_target]],
            tokenizer=norm_tok,
        )["google_bleu"]
        _wer = wer(
            text_target,
            text,
            truth_transform=wer_standardize_contiguous,
            hypothesis_transform=wer_standardize_contiguous,
        )
        _wil = wil(
            text_target,
            text,
            truth_transform=wer_standardize_contiguous,
            hypothesis_transform=wer_standardize_contiguous,
        )
        res = {
            "wer": _wer,
            "wil": _wil,
            "bleu": sentence_bleu_score,
        }
        return res


if __name__ == "__main__":
    audio1, sr = torchaudio.load("../qwq.wav")
    print(
        wer_runner(
            audio1,
            "My young plants require heat, or they would not live; and the pots we are kept in protect us from those cruel wire worms who delight to destroy our roots.",
            sr,
        )
    )
    # audio2 = torch.randn(16000)
    # wespeaker_runner(audio1, audio2, 16000)
