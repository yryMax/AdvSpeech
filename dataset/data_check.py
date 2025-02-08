import os
import re
import torch
import torchaudio


def check_data(root_dir, resample_rate=22050):
    messages = []

    subdirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    if not subdirs:
        messages.append(f"[WARNING] No subdirectories found in '{root_dir}'. Nothing to check.")
        _print_messages(messages)
        return

    #    - base_file：   speaker_{i}.wav
    #    - derived_file：speaker_{i}_xxx.wav
    deriv_pattern = re.compile(r'^(?P<speaker>.+)_(?P<idx>\d+)_(?P<extra>.+)\.wav$')

    # match base file
    for subdir in subdirs:
        speaker_dir = os.path.join(root_dir, subdir)
        wav_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')]

        speaker_files_map = {}

        for fname in wav_files:
            deriv_match = deriv_pattern.match(fname)
            if deriv_match:
                # speaker_{idx}_{extra}.wav
                idx = int(deriv_match.group('idx'))
                if idx not in speaker_files_map:
                    speaker_files_map[idx] = []
                speaker_files_map[idx].append(fname)


        if not speaker_files_map:
            messages.append(f"[ERROR] No valid speaker_i.wav files found in '{speaker_dir}'.")
            continue

        max_i = max(speaker_files_map.keys())
        if max_i < 2:
            messages.append(f"[ERROR] The maximum i is {max_i}, which is < 2 in '{speaker_dir}'.")

        for i_val, deriv_files in speaker_files_map.items():
            base_file = subdir + f"_{i_val}.wav"

            if len(deriv_files) < 2:
                messages.append(
                    f"[ERROR] In '{speaker_dir}', i={i_val} has fewer than 2 derived files "
                    f"({len(deriv_files)} found)."
                )


            base_path = os.path.join(speaker_dir, base_file)
            base_wave, base_sr = _load_and_resample(base_path, resample_rate, messages)
            base_len = base_wave.shape[-1]

            derived_waves = []

            for dfile in deriv_files:
                dpath = os.path.join(speaker_dir, dfile)
                d_wave, d_sr = _load_and_resample(dpath, resample_rate, messages)
                d_len = d_wave.shape[-1]
                if d_len < base_len:
                    messages.append(
                        f"[ERROR] '{dfile}' length ({d_len}) != base file '{base_file}' length ({base_len})."
                    )
                elif d_len > base_len:
                    # cut and save
                    d_wave = d_wave[:, :base_len]
                    torchaudio.save(dpath, d_wave, resample_rate)
                    print(f"[INFO] '{dfile}' length ({d_len}) > base file '{base_file}' length ({base_len}). ")

                derived_waves.append((dfile, d_wave))
            for idx_a in range(len(derived_waves)):
                for idx_b in range(idx_a + 1, len(derived_waves)):
                    file_a, wave_a = derived_waves[idx_a]
                    file_b, wave_b = derived_waves[idx_b]
                    if wave_a.shape == wave_b.shape:
                        if torch.allclose(wave_a, wave_b):
                            messages.append(
                                f"[ERROR] Derived files '{file_a}' and '{file_b}' are bitwise identical in '{speaker_dir}'."
                            )

    _print_messages(messages)


def _load_and_resample(file_path, target_sr, messages):
    if not os.path.isfile(file_path):
        messages.append(f"[ERROR] File '{file_path}' does not exist.")
        return None, None
    try:
        waveform, sr = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        return waveform, target_sr

    except Exception as e:
        messages.append(f"[ERROR] Fail to load '{file_path}': {e}")
        return None, None


def _print_messages(messages):
    if not messages:
        print("All checks passed with no issues!")
    else:
        print("Check results:")
        for msg in messages:
            print(msg)


if __name__ == "__main__":
    check_data(root_dir="/mnt/d/voicedata/LibriTTS/sampled_pair")
