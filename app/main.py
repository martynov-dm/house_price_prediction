from fastapi import FastAPI
from autogluon.tabular import TabularPredictor
import os

# Initialize FastAPI app
app = FastAPI()

# Set the base path for the models
base_path = "./app/model"

# Load the trained AutoGluon models
model_path_ru = os.path.join(base_path, "ru")
model_path_msk = os.path.join(base_path, "msk")
print()

predictor_ru = TabularPredictor.load(model_path_ru)
features_ru = predictor_ru.feature_metadata.get_features()

predictor_msk = TabularPredictor.load(model_path_msk)
features_msk = predictor_msk.feature_metadata.get_features()

print("RU features:", features_ru)
print("MSK features:", features_msk)

# # Dynamically create the HouseFeatures models based on the predictor's features
# feature_dict_ru = {name: (float, ...) for name in feature_names_ru}
# feature_dict_msk = {name: (float, ...) for name in feature_names_msk}

# HouseFeaturesRu = create_model('HouseFeaturesRu', **feature_dict_ru)
# HouseFeaturesMsk = create_model('HouseFeaturesMsk', **feature_dict_msk)


# @app.post("/predict/ru")
# async def predict_price_ru(features: HouseFeaturesRu):
#     try:
#         input_data = pd.DataFrame([features.dict()])
#         prediction = predictor_ru.predict(input_data)
#         return {"predicted_price": float(prediction[0])}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/predict/msk")
# async def predict_price_msk(features: HouseFeaturesMsk):
#     try:
#         input_data = pd.DataFrame([features.dict()])
#         prediction = predictor_msk.predict(input_data)
#         return {"predicted_price": float(prediction[0])}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/")
# async def root():
#     return {"message": "House Price Prediction API"}

# # Note: We've removed the if __name__ == "__main__" block
