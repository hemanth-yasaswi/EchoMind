import numpy as np
import tensorflow as tf

from training.train_esc50 import load_dataset, NUM_CLASSES

# Load test data
_, X_test, _, y_test = load_dataset()
X_test = X_test[..., np.newaxis]
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# Load trained model
model = tf.keras.models.load_model("models/esc50_cnn.keras")

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {acc:.4f}")