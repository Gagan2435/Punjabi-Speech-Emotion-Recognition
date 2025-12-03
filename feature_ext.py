import os
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import parselmouth

import subprocess
import tempfile

from pydub import AudioSegment

# Explicit ffmpeg path (optional since we use subprocess)
AudioSegment.ffmpeg = r"C:\Program Files\Softdeluxe\Free Download Manager\ffmpeg.exe"
AudioSegment.converter = r"C:\Program Files\Softdeluxe\Free Download Manager\ffmpeg.exe"

def convert_to_wav(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".wav":
        return file_path

    try:
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()

        ffmpeg_exe = r"C:\Program Files\Softdeluxe\Free Download Manager\ffmpeg.exe"
        command = [
            ffmpeg_exe,
            "-y",
            "-i", file_path,
            "-ac", "1",
            "-ar", "22050",
            temp_wav.name
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"❌ FFmpeg error converting {file_path}:\n{result.stderr}")
            os.unlink(temp_wav.name)
            return None

        return temp_wav.name

    except Exception as e:
        print(f"❌ Exception during conversion of {file_path}: {e}")
        return None

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        snd = parselmouth.Sound(file_path)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        valid_pitch = pitch_values[pitch_values > 0]
        f0_mean = np.mean(valid_pitch) if len(valid_pitch) > 0 else 0.0

        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

        try:
            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        except Exception as e:
            print(f"⚠️ Jitter extraction failed for {file_path}: {e}")
            jitter = 0.0

        try:
            shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except Exception as e:
            print(f"⚠️ Shimmer extraction failed for {file_path}: {e}")
            shimmer = 0.0

        features = np.concatenate((mfcc, chroma, [zcr, rms, spec_centroid, f0_mean, jitter, shimmer]))
        return features
    except Exception as e:
        print(f"❌ Error extracting features from {file_path}: {e}")
        return None

if __name__ == "__main__":
    base_path = Path(__file__).parent.resolve()
    dataset_folder = base_path / "Punjabi"  # adjust if needed
    output_file = base_path / "feature_ext_output.csv"

    features_list = []
    file_info = []

    print(f"\n🔍 Looking for audio files in {dataset_folder} ...")
    audio_files = list(dataset_folder.rglob("*"))
    audio_files = [f for f in audio_files if f.suffix.lower() in [".m4a", ".mp3", ".wav"]]
    print(f"📁 Found {len(audio_files)} audio files.")

    for idx, file_path in enumerate(audio_files, 1):
        print(f"➡️ [{idx}/{len(audio_files)}] Processing: {file_path}")

        temp_wav_path = convert_to_wav(str(file_path))
        if not temp_wav_path:
            continue

        features = extract_audio_features(temp_wav_path)
        if features is not None:
            features_list.append(features)

            relative_path = file_path.relative_to(dataset_folder)
            parts = relative_path.parts
            participant = parts[0] if len(parts) > 0 else "unknown"
            emotion = parts[1] if len(parts) > 1 else "unknown"

            file_info.append({
                "participant": participant,
                "emotion": emotion,
                "filename": file_path.name,
                "full_path": str(file_path),
                "relative_path": str(relative_path),
            })

        if temp_wav_path != str(file_path) and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

    if features_list:
        feature_names = [f"mfcc_{i+1}" for i in range(13)] + \
                        [f"chroma_{i+1}" for i in range(12)] + \
                        ["zcr", "rms", "spec_centroid", "f0_mean", "jitter", "shimmer"]

        df_features = pd.DataFrame(features_list, columns=feature_names)
        df_info = pd.DataFrame(file_info)
        final_df = pd.concat([df_info, df_features], axis=1)
        final_df.to_csv(output_file, index=False)

        print(f"\n✅ Feature extraction complete!")
        print(f"💾 Saved to: {output_file.resolve()}")
        print(f"📈 Total files processed: {len(final_df)}")
        print(f"🧩 Each file has {len(feature_names)} features (31 in total)")

        print("\n📋 Preview:")
        print(final_df.head())

        print("\n📊 Top participants:")
        print(final_df['participant'].value_counts())

        print("\n🎭 Top emotions:")
        print(final_df['emotion'].value_counts())
    else:
        print("❌ No features extracted. Check dataset structure, FFmpeg installation, or audio file validity.")
