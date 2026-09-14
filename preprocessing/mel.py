import librosa
import numpy as np
import cv2

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
IMG_SIZE = 128

# Fixed dB range used to normalize every clip identically.
# Using a fixed floor/ceiling (instead of each clip's own min/max)
# preserves real loudness differences between classes (a quiet clock
# tick vs. a loud siren) instead of erasing them.
DB_MIN = -80.0
DB_MAX = 0.0


def extract_mel_spectrogram(file_path):
    """
    Converts an audio file into a normalized Mel Spectrogram.

    Returns:
        numpy.ndarray of shape (IMG_SIZE, IMG_SIZE), values in [0, 1]
    """
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    # Convert power to decibel scale, referenced to this clip's own
    # peak power. This part is standard and fine to keep.
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize (time axis) to a fixed square for the CNN input.
    mel_resized = cv2.resize(
        mel_db,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    # Normalize with a FIXED dB range shared by every single clip.
    # Do NOT min-max normalize per-clip here -- that was the previous
    # bug: it stretches every spectrogram (quiet or loud) to fill
    # exactly [0, 1], throwing away real inter-class loudness cues.
    mel_clipped = np.clip(mel_resized, DB_MIN, DB_MAX)
    mel_normalized = (mel_clipped - DB_MIN) / (DB_MAX - DB_MIN)

    return mel_normalized.astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sample = "datasets/esc50/audio/1-137-A-32.wav"
    spectrogram = extract_mel_spectrogram(sample)

    plt.figure(figsize=(5, 5))
    plt.imshow(spectrogram, origin="lower", aspect="auto", cmap="magma")
    plt.colorbar()
    plt.title("Mel Spectrogram")
    plt.show()