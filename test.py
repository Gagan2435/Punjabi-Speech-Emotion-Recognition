import speech_recognition as sr
import librosa
import numpy as np
import pandas as pd
import joblib
import pickle

# Load model, scaler, and label encoder
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Feature extraction
def extract_features(file_path):
    y, sr_rate = librosa.load(file_path, sr=22050)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr_rate).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_rate).T, axis=0)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0
    jitter = np.std(f0) / f0_mean if f0_mean != 0 else 0
    shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) != 0 else 0

    features = np.hstack([mfccs, chroma, zcr, rms, spec_centroid, f0_mean, jitter, shimmer])
    return features.reshape(1, -1)

feature_columns = [f"mfcc_{i+1}" for i in range(13)] + \
                  [f"chroma_{i+1}" for i in range(12)] + \
                  ["zcr", "rms", "spec_centroid", "f0_mean", "jitter", "shimmer"]

# Record and test
r = sr.Recognizer()
try:
    with sr.Microphone() as source:
        print("🎤 Speak now (Punjabi)...")
        audio = r.listen(source)

    with open("live_input.wav", "wb") as f:
        f.write(audio.get_wav_data())

    print("✅ Recording complete!")

    try:
        text = r.recognize_google(audio, language="pa-IN")
        print(f"✅ Transcribed Text: {text}")
    except:
        print("❌ Could not transcribe audio.")

    # Extract and prepare features
    features = extract_features("live_input.wav")
    features_df = pd.DataFrame(features, columns=feature_columns)
    features_scaled = scaler.transform(features_df)

    print(f"✅ Feature shape: {features_scaled.shape}")
    print(f"✅ Feature preview: {features_scaled[0][:5]}")

    # Predict
    pred_numeric = model.predict(features_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_numeric])[0]
    pred_proba = model.predict_proba(features_scaled)[0]

    print(f"\n🔮 Predicted Emotion: {pred_label}")
    print("📊 Prediction Probabilities:")
    for label, prob in zip(label_encoder.classes_, pred_proba):
        print(f"{label}: {prob:.4f}")

except Exception as e:
    print(f"⚠️ Error: {e}")

