"""Test a trained model locally without needing Telegram."""
import argparse
from pathlib import Path
from waste_ai.inference import WasteClassifier, format_result


def main():
    p = argparse.ArgumentParser(description="Predict the class of one image")
    p.add_argument("image", type=Path)
    p.add_argument("--model-path", type=Path)
    args = p.parse_args()
    if not args.image.is_file():
        p.error(f"No such image: {args.image}")
    classifier = WasteClassifier(args.model_path)
    print(format_result(classifier.predict_bytes(args.image.read_bytes())))


if __name__ == "__main__":
    main()
