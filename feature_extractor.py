import numpy as np
import librosa
# from src.config import IMG_SIZE

SR = 8000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 128
IMG_SIZE=(64,64)
def extract_mel_spectrogram(file_path):
    """
    Convert a WAV file to a normalized Mel spectrogram (64x64x1)
    """

    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=SR)

        # 🚨 Safety: empty or corrupt audio
        if y is None or len(y) == 0:
            print(f"❌ Empty audio file: {file_path}")
            return None

        # Generate mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        # 🚨 Safety: avoid division by zero
        if mel_db.max() == mel_db.min():
            print(f"❌ Invalid mel spectrogram (flat signal): {file_path}")
            return None

        # Normalize to 0–1
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

        # Pad or trim to IMG_SIZE
        if mel_db.shape[1] < IMG_SIZE[1]:
            pad_width = IMG_SIZE[1] - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :IMG_SIZE[1]]

        # Final shape check
        if mel_db.shape != IMG_SIZE:
            print(f"❌ Shape mismatch: {mel_db.shape} expected {IMG_SIZE}")
            return None

        return mel_db[..., np.newaxis]  # (64,64,1)

    except Exception as e:
        print(f"❌ Feature extraction error for {file_path}: {e}")
        return None
