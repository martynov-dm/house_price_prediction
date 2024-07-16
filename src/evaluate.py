import sys
import os
import json
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def load_model(model_path):
    return TabularPredictor.load(model_path)


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def plot_actual_vs_predicted(y_true, y_pred, output_dir):
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [
             y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
    plt.close()


def plot_residuals(y_true, y_pred, output_dir):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_distribution.png'))
    plt.close()


def main(data_path, model_path, region):
    # Load data and model
    data = load_data(data_path)
    model = load_model(model_path)

    # Assuming the target column is named 'price'
    X = data.drop('price', axis=1)
    y = data['price']

    # Evaluate model
    metrics = evaluate_model(model, X, y)

    # Save metrics
    os.makedirs('metrics', exist_ok=True)
    with open(f'metrics/{region}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Generate and save plots
    os.makedirs(f'plots/{region}', exist_ok=True)
    predictions = model.predict(X)
    plot_actual_vs_predicted(y, predictions, f'plots/{region}')
    plot_residuals(y, predictions, f'plots/{region}')

    print(f"Evaluation complete for {region}. Metrics and plots saved.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python evaluate.py <data_path> <model_path> <region>")
        sys.exit(1)

    data_path = sys.argv[1]
    model_path = sys.argv[2]
    region = sys.argv[3]

    main(data_path, model_path, region)
