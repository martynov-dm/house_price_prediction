import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from box import ConfigBox
from ruamel.yaml import YAML
from dvclive import Live


def load_data(file_path):
    return pd.read_csv(file_path)


def load_model(model_path):
    return TabularPredictor.load(model_path)


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    mape = mean_absolute_percentage_error(y, predictions)
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
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


def evaluate_region(region, params):
    script_dir = Path(__file__).parent.absolute()
    data_path = script_dir.parent / 'data' / 'prepared' / region / 'test.csv'
    model_path = script_dir.parent / 'models' / region

    # Load data and model
    data = load_data(data_path)
    model = load_model(model_path)

    # Assuming the target column is named 'log_price'
    X = data.drop('log_price', axis=1)
    y = data['log_price']

    # Evaluate model
    metrics = evaluate_model(model, X, y)

    # Log metrics and plots using DVCLive
    with Live(dir=f"dvclive_{region}", save_dvc_exp=True) as live:
        for metric, value in metrics.items():
            live.log_metric(metric, value)

        predictions = model.predict(X)

        if params.evaluate.plot_perfect_predictions:
            plot_actual_vs_predicted(y, predictions, live.dir)
            live.log_image('actual_vs_predicted.png')

        if params.evaluate.plot_residuals:
            plot_residuals(y, predictions, live.dir)
            live.log_image('residual_distribution.png')

    print(f"Evaluation complete for {region}. Metrics and plots saved.")


def main():
    # Load parameters
    yaml = YAML(typ="safe")
    params_path = Path("params.yaml")
    if not params_path.exists():
        raise FileNotFoundError(
            f"params.yaml not found in {params_path.absolute()}")

    params = ConfigBox(yaml.load(params_path.open(encoding="utf-8")))

    # Evaluate both models
    evaluate_region('msk', params)
    evaluate_region('ru', params)


if __name__ == "__main__":
    main()
