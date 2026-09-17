import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Activation,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dropout,
    Dense,
    Add,
)
from tensorflow.keras.models import Model

# Access preprocessing folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.mel import extract_mel_spectrogram, IMG_SIZE

CSV_PATH = "datasets/esc50/meta/esc50.csv"
AUDIO_PATH = "datasets/esc50/audio"
CACHE_DIR = "datasets/esc50/cache"
TEST_FOLD = 5
NUM_CLASSES = 50


def load_dataset():
    """
    Loads ESC-50 audio, converts each clip to a mel spectrogram, and
    splits into train/test using fold 5 as the held-out test set
    (ESC-50's official evaluation protocol).

    Extracted features are cached to disk as .npy files so repeated
    runs while you're tuning the model don't have to re-run librosa
    on all 2000 clips every single time.

    IMPORTANT: if you ever change extract_mel_spectrogram() (e.g. the
    normalization logic), delete the CACHE_DIR folder first, or you'll
    keep training on stale features.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_paths = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "y_test": os.path.join(CACHE_DIR, "y_test.npy"),
    }

    if all(os.path.exists(p) for p in cache_paths.values()):
        print("Loading cached mel spectrograms...")
        X_train = np.load(cache_paths["X_train"])
        X_test = np.load(cache_paths["X_test"])
        y_train = np.load(cache_paths["y_train"])
        y_test = np.load(cache_paths["y_test"])
        return X_train, X_test, y_train, y_test

    print("Extracting mel spectrograms (first run -- this takes a while)...")
    df = pd.read_csv(CSV_PATH)

    train_df = df[df["fold"] != TEST_FOLD]
    test_df = df[df["fold"] == TEST_FOLD]

    X_train, y_train = [], []
    X_test, y_test = [], []

    for _, row in train_df.iterrows():
        file_path = os.path.join(AUDIO_PATH, row["filename"])
        X_train.append(extract_mel_spectrogram(file_path))
        y_train.append(row["target"])

    for _, row in test_df.iterrows():
        file_path = os.path.join(AUDIO_PATH, row["filename"])
        X_test.append(extract_mel_spectrogram(file_path))
        y_test.append(row["target"])

    X_train, X_test = np.array(X_train), np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)

    np.save(cache_paths["X_train"], X_train)
    np.save(cache_paths["X_test"], X_test)
    np.save(cache_paths["y_train"], y_train)
    np.save(cache_paths["y_test"], y_test)

    return X_train, X_test, y_train, y_test


def spec_augment(mel, num_freq_masks=1, num_time_masks=1,
                  freq_mask_width=8, time_mask_width=8):
    """
    Lightweight SpecAugment: randomly zeroes a few frequency bands and
    time slices. Applied only to TRAINING data. With only ~26 clips
    per class after the train/val split, this matters a lot -- it
    stops the CNN from memorizing exact spectrogram patterns.
    """
    mel = mel.copy()
    h, w = mel.shape[:2]

    for _ in range(num_freq_masks):
        f = np.random.randint(1, freq_mask_width)
        f0 = np.random.randint(0, max(1, h - f))
        mel[f0:f0 + f, :, :] = 0.0

    for _ in range(num_time_masks):
        t = np.random.randint(1, time_mask_width)
        t0 = np.random.randint(0, max(1, w - t))
        mel[:, t0:t0 + t, :] = 0.0

    return mel


def make_train_dataset(X, y, batch_size=32):
    """tf.data pipeline that re-applies random augmentation every epoch."""
    def augment_map(x, label):
        x = tf.numpy_function(spec_augment, [x], tf.float32)
        x.set_shape([IMG_SIZE, IMG_SIZE, 1])
        return x, label

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    ds = ds.map(augment_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def residual_block(x, filters):
    shortcut = x

    x = Conv2D(filters, (3, 3), padding="same",
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, (3, 3), padding="same",
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    return x


def build_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = Conv2D(32, (3, 3), padding="same",
               kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding="same",
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), padding="same",
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # NEW: Residual block
    x = residual_block(x, 128)

    x = MaxPooling2D((2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu",
              kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.4)(x)

    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    return model

def main():
    print("Loading ESC-50 dataset...")
    X_train, X_test, y_train, y_test = load_dataset()

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Keep integer labels around for stratified splitting, before
    # converting to one-hot.
    y_train_int = y_train.copy()
    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))
    print("Input shape:", X_train.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train_int,
    )

    # Create the models/ directory BEFORE fit() -- ModelCheckpoint
    # will crash on epoch 1 if this folder doesn't already exist.
    os.makedirs("models", exist_ok=True)

    model = build_model()
    model.summary()

    train_ds = make_train_dataset(X_train, y_train, batch_size=32)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=15,
        restore_best_weights=True,
    )

    checkpoint = ModelCheckpoint(
        "models/esc50_cnn.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print("Model saved to models/esc50_cnn.keras")


if __name__ == "__main__":
    main()