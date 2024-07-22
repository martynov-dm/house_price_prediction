from autogluon.tabular import TabularPredictor, TabularDataset
import pandas as pd
from pathlib import Path

script_dir = Path(__file__).parent.absolute()

# Load the predictor
predictor_ru = TabularPredictor.load('./models/ru', verbosity=4)

# Persist the models
predictor_ru.persist()

# Load test data
test_csv_path = script_dir / 'data' / 'prepared' / 'ru' / 'test.csv'
test_data = pd.read_csv(test_csv_path)
test_data = TabularDataset(test_data)

# Make predictions
predictions = predictor_ru.predict(test_data)
