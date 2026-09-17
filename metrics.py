import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from training.train_esc50 import load_dataset, NUM_CLASSES

# -----------------------
# Load dataset
# -----------------------
_, X_test, _, y_test = load_dataset()

X_test = X_test[..., np.newaxis]

# Integer labels
y_true = y_test.copy()

# One-hot labels for evaluation
y_test_cat = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# -----------------------
# Load model
# -----------------------
model = tf.keras.models.load_model("models/esc50_cnn.keras")

# -----------------------
# Predictions
# -----------------------
pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(pred_probs, axis=1)

# -----------------------
# Accuracy
# -----------------------
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {accuracy:.4f}")

# -----------------------
# Class names
# -----------------------
df = pd.read_csv("datasets/esc50/meta/esc50.csv")
class_names = (
    df.sort_values("target")["category"]
      .drop_duplicates()
      .tolist()
)

# -----------------------
# Classification report
# -----------------------
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=3,
)

os.makedirs("results", exist_ok=True)

with open("results/classification_report.txt", "w") as f:
    f.write(report)

print("\nClassification Report:\n")
print(report)

# -----------------------
# Confusion matrix
# -----------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 14))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

disp.plot(
    cmap="Blues",
    include_values=False,
    xticks_rotation=90,
    colorbar=True,
)

plt.title("EchoMind – ESC-50 Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("True Label", fontsize=11)

plt.xticks(fontsize=6)
plt.yticks(fontsize=6)

plt.tight_layout()
plt.savefig(
    "results/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved confusion matrix to results/confusion_matrix.png")
print("Saved classification report to results/classification_report.txt")