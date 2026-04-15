# House Price Prediction

This project demonstrates a simple house price prediction pipeline using Python, scikit-learn, and a linear regression model.

## Project Structure

- `data/housing.csv` - sample housing dataset
- `house_price_analysis.ipynb` - optional notebook for analysis and model training
- `src/preprocess.py` - data loading and preprocessing utilities
- `src/train.py` - model training and evaluation script
- `src/predict.py` - model inference utilities
- `house_price_model.pkl` - saved regression model artifact
- `reports/model_evaluation.txt` - model evaluation summary
- `house_price_prediction.py` - command-line interface to train and predict
- `correlation_heatmap.png` - generated correlation heatmap plot
- `predictions_scatter.png` - generated actual vs predicted scatter plot

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Train the model:

```bash
python house_price_prediction.py train
```

Predict using the saved model:

```bash
python house_price_prediction.py predict --input data/housing.csv
```

## Notes

- Update `data/housing.csv` with the full dataset for realistic training.
- The sample dataset included is small and primarily for demonstration.
