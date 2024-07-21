import pandas as pd
import logging
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
import traceback
from box import ConfigBox
from ruamel.yaml import YAML
from pathlib import Path
from dvclive import Live
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def train_model(model_type, presets, time_limit):
    script_dir = Path(__file__).parent.absolute()

    train_csv_path = script_dir.parent / 'data' / \
        'prepared' / model_type / 'train.csv'
    test_csv_path = script_dir.parent / 'data' / 'prepared' / model_type / 'test.csv'
    output_model_path = script_dir.parent / 'models' / model_type

    if not train_csv_path.exists():
        raise FileNotFoundError(
            f"Prepared data file not found: {train_csv_path}")

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

    # Create a unique Live object for each model type
    with Live(dir=f"dvclive_{model_type}", save_dvc_exp=True) as live:
        live.log_param("presets", presets[0])
        live.log_param("model_type", model_type)

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

        evaluation_results = autogluon_automl.evaluate(test_data)
        logging.info(f"Evaluation results for {model_type} model:")
        logging.info(evaluation_results)

        live.log_artifact(
            str(output_model_path),
            type="model",
            name=model_type,
            desc=f"House price prediction for {model_type} model",
            labels=["AutoGluon", model_type],
        )

        y_pred = autogluon_automl.predict(test_data)
        y_true = test_data['log_price']

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        live.log_metric("RMSE", rmse)

        y_true_exp = np.expm1(y_true)
        y_pred_exp = np.expm1(y_pred)
        mae_rub = mean_absolute_error(y_true_exp, y_pred_exp)
        live.log_metric("MAE_RUB", mae_rub)

        mape = np.mean(np.abs((y_true_exp - y_pred_exp) / y_true_exp)) * 100
        live.log_metric("MAPE", mape)

        r2 = r2_score(y_true, y_pred)
        live.log_metric("R2", r2)


def main():
    yaml = YAML(typ="safe")
    params_path = Path("params.yaml")
    if not params_path.exists():
        logging.error(f"params.yaml not found in {params_path.absolute()}")
        sys.exit(1)

    params = ConfigBox(yaml.load(params_path.open(encoding="utf-8")))

    # Train MSK model
    logging.info("Training Moscow model")
    train_model('msk', params.train_msk.presets, params.train_msk.time_limit)

    # Train RU model
    logging.info("Training RU model")
    train_model('ru', params.train_ru.presets, params.train_ru.time_limit)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Script failed with error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
