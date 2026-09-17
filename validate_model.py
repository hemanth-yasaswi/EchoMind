import os
import pandas as pd
import numpy as np
import tensorflow.keras.models
from inference.predict import AudioPredictor


def main():
    # Load the trained model
    model_path = "models/esc50_cnn.keras"
    try:
        model = tensorflow.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    # Load ESC-50 metadata
    csv_path = "datasets/esc50/meta/esc50.csv"
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV from {csv_path}: {e}")
        return

    # Filter fold 5 and select 5 random samples
    test_df = df[df["fold"] == 5]
    if len(test_df) < 5:
        print("Not enough samples in fold 5. Found only", len(test_df))
        return

    samples = test_df.sample(n=5).reset_index(drop=True)

    # Initialize predictor
    predictor = AudioPredictor()

    # Process each sample
    correct_predictions = 0
    for idx, row in samples.iterrows():
        filename = row["filename"]
        true_class_idx = row["target"]
        true_class_label = row["category"]  # Assuming 'category' column exists in CSV

        # Get prediction
        try:
            predictions = predictor.predict(os.path.join("datasets/esc50/audio", filename))
        except Exception as e:
            print(f"Error predicting for {filename}: {e}")
            continue

        # Extract top-1 prediction
        predicted_class_idx, predicted_class_label, confidence = predictions[0]

        # Determine correctness
        is_correct = predicted_class_idx == true_class_idx

        # Print results
        print(f"Filename: {filename}")
        print(f"True Class Index: {true_class_idx}")
        print(f"True Class Label: {true_class_label}")
        print(f"Predicted Class Index: {predicted_class_idx}")
        print(f"Predicted Class Label: {predicted_class_label}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Correct: {'Yes' if is_correct else 'No'}")
        print("-" * 60)

        # Update accuracy
        if is_correct:
            correct_predictions += 1

    # Print final accuracy
    accuracy = correct_predictions / len(samples)
    print(f"Top-1 Accuracy on 5 samples from fold 5: {accuracy:.2f} ({correct_predictions}/{len(samples)})")


if __name__ == "__main__":
    main()
