🎤 Punjabi Speech Emotion Recognition – ELC Internship (Thapar Institute)

This project is part of the Voice Assistant System developed during the ELC Internship at Thapar Institute of Engineering & Technology, under the guidance of Dr. Aditi.
My contribution focused on building a Punjabi Speech Emotion Detection Model that identifies emotions directly from speech patterns, not the spoken words.

📚 Dataset Collection 

The dataset used in this project was completely self-collected by me and my team during the ELC Internship.
We conducted a structured data collection process inside Thapar University, where:

✔️ 15 student volunteers participated

✔️ Each participant spoke 15 Punjabi sentences

✔️ Each sentence was recorded in 5 different emotions:

😡 Anger

😢 Sad

😀 Happy

😐 Neutral

😨 Fear

✔️ All recordings were supervised and documented under Dr. Aditi Mam

Total dataset size created by us:

15 participants × 15 sentences × 5 emotions = 1125 audio samples

This dataset is unique, real, and built entirely by our team, making it one of the first structured Punjabi emotion datasets of this type.

❗Why the dataset cannot be uploaded

Since the recordings contain personal voice data of Thapar students, collected internally, the dataset is private and cannot be shared on GitHub due to:

Privacy concerns

Ethical data-use restrictions

No public release rights

Only the features and trained model are uploaded .

🔍 Project Overview

The goal of this module was to detect human emotions from Punjabi audio clips using acoustic features.
The system analyzes tone, pitch, energy, MFCC, and other speech patterns to classify emotion.

🧠 Machine Learning Model

Random Forest Classifier trained on extracted and scaled features.
Includes label encoding, standardization, stratified train/test split, and confusion matrix evaluation.

🎧 Feature Extraction

Uses librosa to extract:

MFCC (13 coefficients)

Chroma

ZCR

RMS

Spectral Centroid

F0 (fundamental frequency)

Jitter / Shimmer

📊 Model Evaluation

Classification Report

Confusion Matrix

Probability Distribution

Saved models:

rf_model.pkl

scaler.pkl

label_encoder.pkl

🧪 Live Emotion Testing From Microphone

Complete code provided to:

Record Punjabi speech

Extract audio features

Scale features

Predict emotion

Show probabilities

Very useful for real-time emotion-based voice assistant systems.

🙋‍♂️ My Contribution

Dataset collection & organization

Feature engineering

Model training (Random Forest)

Evaluation & confusion matrix

Real-time microphone emotion testing

Saving and optimizing model artifacts

🔮 Future Scope

Increase dataset variety (age groups, accents)

Add new emotions (surprise, disgust, uncertainty)

Replace RF with CNN, LSTM, wav2vec2.0

Integrate into a full multilingual voice assistant
