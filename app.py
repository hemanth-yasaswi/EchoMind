import argparse
from inference.predict import AudioPredictor


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Predict audio class from a file.")
    parser.add_argument("--audio", required=True, help="Path to the audio file to predict.")
    args = parser.parse_args()

    # Initialize the predictor
    try:
        predictor = AudioPredictor()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Attempt to make prediction
    try:
        predictions = predictor.predict(args.audio)
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return

    # Print top-5 predictions
    print("Top 5 Predictions:")
    for label, score in predictions:
        print(f"{label.replace('_', ' '):20} {score*100:6.2f}%")


if __name__ == "__main__":
    main()
