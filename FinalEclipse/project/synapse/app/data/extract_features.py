import numpy as np
import librosa

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050)

        # Ensure audio is valid
        if len(audio) == 0:
            return None

        # MFCC (40)
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)

        # Chroma (12)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)

        # Spectral Contrast (7)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)

        # ZCR (1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))

        features = np.hstack([mfcc, chroma, contrast, zcr])  # total = 60

        return features

    except Exception as e:
        print("❌ Error:", file_path)
        print("Reason:", e)
        return None