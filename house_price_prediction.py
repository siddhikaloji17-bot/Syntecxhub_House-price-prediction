from argparse import ArgumentParser
from pathlib import Path

from src.predict import load_model, load_input_data, predict
from src.train import train_model

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "housing.csv"
MODEL_PATH = ROOT / "models" / "linear_regression_model.pkl"
REPORT_PATH = ROOT / "reports" / "model_evaluation.txt"


def parse_args():
    parser = ArgumentParser(description="House Price Prediction CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the regression model")
    train_parser.add_argument("--data", default=DATA_PATH, help="Path to the training CSV file")
    train_parser.add_argument("--output", default=MODEL_PATH, help="Path to save the trained model")
    train_parser.add_argument("--report", default=REPORT_PATH, help="Path to save evaluation report")

    predict_parser = subparsers.add_parser("predict", help="Predict house prices")
    predict_parser.add_argument("--model", default=MODEL_PATH, help="Path to the trained model")
    predict_parser.add_argument("--input", default=DATA_PATH, help="Path to input CSV file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "train":
        train_model(args.data, args.output, args.report)
    elif args.command == "predict":
        model = load_model(args.model)
        input_df = load_input_data(args.input)
        predictions = predict(model, input_df)
        print("Predictions for input rows:")
        for idx, value in enumerate(predictions, start=1):
            print(f"Row {idx}: {value:.2f}")
