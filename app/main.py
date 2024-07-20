from fastapi import FastAPI, HTTPException
from autogluon.tabular import TabularPredictor
import os
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.data_transformers import prepare_features_msk, prepare_features_ru
from app.predict import predict_price_msk, predict_price_ru
from .models import RuInput, MskInput

app = FastAPI(root_path="/api")

current_dir = os.path.abspath(os.getcwd())

base_path = os.path.join(current_dir, "models")

model_path_ru = os.path.join(base_path, "ru")
model_path_msk = os.path.join(base_path, "msk")

predictor_ru = TabularPredictor.load(model_path_ru, verbosity=4)
predictor_msk = TabularPredictor.load(model_path_msk, verbosity=4)


origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict/ru")
async def predict_ru(input_data: RuInput):

    ru_features = prepare_features_ru(input_data)

    try:
        ru_price = predict_price_ru(predictor_ru, ru_features)
        return {"predicted_price": ru_price}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/msk")
async def predict_msk(input_data: MskInput):

    msk_features = prepare_features_msk(input_data)

    try:
        msk_price = predict_price_msk(predictor_msk, msk_features)
        return {"predicted_price": msk_price}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=True)
