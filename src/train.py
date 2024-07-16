import sys
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def train_model(input_data_file, output_model_path, model_type):
    try:
        logging.info(f"Starting model training for {model_type}")
        logging.info(f"Loading data from {input_data_file}")
        df_working = pd.read_csv(input_data_file)
        logging.info(f"Data loaded. Shape: {df_working.shape}")

        df_working = TabularDataset(df_working)
        logging.info("Converted to TabularDataset")

        logging.info("Splitting data into train and test sets")
        train_data, test_data = train_test_split(
            df_working, test_size=0.2, random_state=42)
        logging.info(
            f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")

        model_params = {
            'path': output_model_path,
            'label': 'log_price',
            'problem_type': 'regression',
            'eval_metric': 'mean_squared_error',
            'verbosity': 4

        }
        logging.info(f"Model parameters: {model_params}")

        if model_type == 'msk':
            logging.info("Training Moscow model")
        elif model_type == 'ru':
            logging.info("Training Russia-wide model")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logging.info("Initializing TabularPredictor")
        autogluon_automl = TabularPredictor(**model_params)

        logging.info("Initializing AutoMLPipelineFeatureGenerator")
        auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()

        logging.info("Starting model training")
        autogluon_automl.fit(
            train_data=train_data, feature_generator=auto_ml_pipeline_feature_generator)
        logging.info("Model training completed")

        logging.info("Evaluating model")
        evaluation_results = autogluon_automl.evaluate(test_data)
        logging.info(f"Evaluation results for {model_type} model:")
        logging.info(evaluation_results)

        logging.info(f"Model training for {model_type} completed successfully")
    except Exception as e:
        logging.error(f"An error occurred during model training: {str(e)}")
        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    if len(sys.argv) != 4:
        logging.error(
            "Usage: python train.py <input_data_file> <output_model_path> <model_type>")
        sys.exit(1)

    input_data_file = sys.argv[1]
    output_model_path = sys.argv[2]
    model_type = sys.argv[3]

    try:
        train_model(input_data_file, output_model_path, model_type)
    except Exception as e:
        logging.error(f"Script failed with error: {str(e)}")
        sys.exit(1)
