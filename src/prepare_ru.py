import pandas as pd
import logging
import wandb
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
import traceback
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import sys
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
PARAMS = {
    'msk': {
        'presets': ['medium_quality', 'optimize_for_deployment'],
        'time_limit': 900  # 15 minutes
    },
    'ru': {
        'presets': ['medium_quality', 'optimize_for_deployment'],
        'time_limit': 900  # 15 minutes
    }
}


def evaluate_and_log_metrics(y_true, y_pred, prefix=""):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    mae_rub = mean_absolute_error(y_true_exp, y_pred_exp)
    mape = np.mean(np.abs((y_true_exp - y_pred_exp) / y_true_exp)) * 100
    r2 = r2_score(y_true, y_pred)

    wandb.log({
        f"{prefix}RMSE": rmse,
        f"{prefix}MAE_RUB": mae_rub,
        f"{prefix}MAPE": mape,
        f"{prefix}R2": r2
    })

    return {
        "RMSE": rmse,
        "MAE_RUB": mae_rub,
        "MAPE": mape,
        "R2": r2
    }


def safe_remove_dir(path):
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
        logging.info(f"Removed existing directory: {path}")


def train_model(model_type, presets, time_limit):
    script_dir = Path(__file__).parent.absolute()

    train_csv_path = script_dir.parent / 'data' / \
        'prepared' / model_type / 'train.csv'
    test_csv_path = script_dir.parent / 'data' / 'prepared' / model_type / 'test.csv'
    output_model_path = script_dir.parent / 'models' / model_type

    if not train_csv_path.exists():
        raise FileNotFoundError(
            f"Prepared data file not found: {train_csv_path}")

    # Remove existing model directory to ensure a clean start
    safe_remove_dir(output_model_path)

    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)

    train_data = TabularDataset(train_data)
    test_data = TabularDataset(test_data)

    model_params = {
        'path': str(output_model_path),
        'label': 'log_price',
        'problem_type': 'regression',
        'eval_metric': 'mean_squared_error',
        'verbosity': 2
    }

    # Initialize wandb run
    with wandb.init(project="house_price_prediction", name=f"{model_type}_model", config={
        "presets": presets,
        "model_type": model_type,
        "time_limit": time_limit
    }) as run:
        autogluon_automl = TabularPredictor(**model_params)
        auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator(
            enable_numeric_features=True,
            enable_categorical_features=True,
            enable_datetime_features=True,
            enable_text_special_features=False,
            enable_text_ngram_features=False,
            enable_raw_text_features=False,
            enable_vision_features=False,
        )

        kwargs = {
            'excluded_model_types': ['FASTAI', 'NN_TORCH']}

        autogluon_automl.fit(
            train_data=train_data,
            feature_generator=auto_ml_pipeline_feature_generator,
            presets=presets,
            time_limit=time_limit,
            **kwargs)

        # Evaluate initial model
        y_pred = autogluon_automl.predict(test_data)
        y_true = test_data['log_price']
        initial_metrics = evaluate_and_log_metrics(
            y_true, y_pred, prefix="initial_")

        logging.info(f"Initial evaluation results for {model_type} model:")
        logging.info(initial_metrics)

        # Attempt refit_full
        try:
            logging.info(f"Refitting {model_type} model on full dataset...")
            autogluon_automl.refit_full()

            # Evaluate refitted model
            y_pred_refit = autogluon_automl.predict(test_data)
            refit_metrics = evaluate_and_log_metrics(
                y_true, y_pred_refit, prefix="refit_")

            logging.info(f"Refit evaluation results for {model_type} model:")
            logging.info(refit_metrics)
        except Exception as e:
            logging.error(
                f"Error during refit_full for {model_type} model: {str(e)}")
            logging.info("Proceeding with the initial model...")

        # Log model as an artifact
        model_artifact = wandb.Artifact(
            name=f"{model_type}_model",
            type="model",
            description=f"House price prediction model for {model_type}",
        )
        model_artifact.add_dir(str(output_model_path))
        run.log_artifact(model_artifact)


def main():
    # Login to wandb
    wandb.login()

    # Train MSK model
    logging.info("Training Moscow model")
    train_model('msk', PARAMS['msk']['presets'], PARAMS['msk']['time_limit'])

    # Train RU model
    logging.info("Training RU model")
    train_model('ru', PARAMS['ru']['presets'], PARAMS['ru']['time_limit'])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script failed with error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
