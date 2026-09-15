import numpy as np
import tensorflow.keras.models
from preprocessing.mel import extract_mel_spectrogram


class AudioPredictor:
    def __init__(self, model_path="models/esc50_cnn.keras", labels_path="esc50_classes.txt"):
        """
        Initialize the predictor with model and label mappings.
        """
        self.model = self._load_model(model_path)
        self.class_labels = self._load_labels(labels_path)

    def _load_model(self, model_path):
        """
        Load the pre-trained Keras model.
        """
        try:
            model = tensorflow.keras.models.load_model(model_path)
            return model
        except Exception as e:
            raise FileNotFoundError(f"Model not found at {model_path}. Error: {e}")

    def _load_labels(self, labels_path):
        """
        Load class labels from a text file.
        """
        try:
            with open(labels_path, "r") as f:
                labels = [line.strip() for line in f.readlines()]
            return labels
        except Exception as e:
            raise FileNotFoundError(f"Labels file not found at {labels_path}. Error: {e}")

    def predict(self, audio_file_path):
        """
        Predict the top-5 classes for the given audio file.

        Returns:
            List of tuples (class_name, confidence_score)
        """
        # Extract and preprocess the spectrogram
        spectrogram = extract_mel_spectrogram(audio_file_path)
        spectrogram = np.expand_dims(spectrogram, axis=(0, -1))  # Add batch and channel dimensions

        # Get model predictions
        predictions = self.model.predict(spectrogram)[0]  # Remove batch dimension

        # Get top-5 indices and sort by confidence
        top_indices = np.argsort(predictions)[-5:][::-1]
        top_classes = [(self.class_labels[i], float(predictions[i])) for i in top_indices]

        return top_classes
