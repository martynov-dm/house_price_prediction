from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd
from app.models import MskFeatures, RuFeatures
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_price_ru(predictor: TabularPredictor, features: RuFeatures) -> float:

    df = pd.DataFrame([features])
    logger.info("Starting prediction ru")

    prediction = predictor.predict(df)
    return float(np.expm1(prediction[0]))


def predict_price_msk(predictor: TabularPredictor, features: MskFeatures) -> float:
    df = pd.DataFrame([features])

    logger.info("Starting prediction msk")
    try:
        prediction = predictor.predict(df)
        logger.info("Prediction completed successfully")
        logger.info(f"Raw prediction: {prediction}")
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

    result = float(np.expm1(prediction[0]))
    logger.info(f"Final prediction (after expm1): {result}")

    return result
