from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load model and scaler
model_1 = joblib.load("dyslexia_predictor_model.joblib")
model_2 = joblib.load("child_dyslexia_xgb_model.pkl")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your Node.js domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define expected features from Node.js

model = joblib.load("dyslexia_rf_model.pkl")

app = FastAPI()


class PredictRequest(BaseModel):
    age: int
    phonemeMatching_score: int
    phonemeMatching_time: int
    phonemeMatching_errorsCount: int
    letterRecognition_score: int
    letterRecognition_time: int
    letterRecognition_errorsCount: int
    attention_score: int
    attention_time: int
    attention_errorsCount: int
    patternMemory_score: int
    patternMemory_time: int
    patternMemory_errorsCount: int
    reading_pronunciationAccuracy: int
    reading_readingSpeedWpm: int
    reading_timeTaken: int
    reading_totalErrors: int
    reading_totalScore: int
    reading_readingFluency: int
    letterReversalCount: int
    ageStartedReading: int
    familyHistoryOfDyslexia: bool


@app.post("/predict-english-test")
def predict_child(data: PredictRequest):
    features = [[
        data.age,
        data.phonemeMatching_score,
        data.phonemeMatching_time,
        data.phonemeMatching_errorsCount,
        data.letterRecognition_score,
        data.letterRecognition_time,
        data.letterRecognition_errorsCount,
        data.attention_score,
        data.attention_time,
        data.attention_errorsCount,
        data.patternMemory_score,
        data.patternMemory_time,
        data.patternMemory_errorsCount,
        data.reading_pronunciationAccuracy,
        data.reading_readingSpeedWpm,
        data.reading_timeTaken,
        data.reading_totalErrors,
        data.reading_totalScore,
        data.reading_readingFluency,
        data.letterReversalCount,
        data.ageStartedReading,
        int(data.familyHistoryOfDyslexia)
    ]]

    prediction = model.predict(features)[0]
    print("Model Prediction for English speaker: ", prediction)
    return {
        "prediction": int(prediction),
        "label": "Dyslexic" if prediction == 1 else "Non-Dyslexic"
    }




class TaskBasedInput(BaseModel):
    age: int
    phonemeMatching_score: int
    phonemeMatching_time: float
    phonemeMatching_errors: int
    letterRecognition_score: int
    letterRecognition_time: float
    letterRecognition_errors: int
    attention_score: int
    attention_time: float
    attention_errors: int
    patternMemory_score: int
    patternMemory_time: float
    patternMemory_errors: int

# 🔮 Define prediction route
@app.post("/predict-child-test")
def predict(data: TaskBasedInput):
    # 👇 Prepare input in same order as training
    input_array = np.array([[
        data.age,
        data.phonemeMatching_score,
        data.phonemeMatching_time,
        data.phonemeMatching_errors,
        data.letterRecognition_score,
        data.letterRecognition_time,
        data.letterRecognition_errors,
        data.attention_score,
        data.attention_time,
        data.attention_errors,
        data.patternMemory_score,
        data.patternMemory_time,
        data.patternMemory_errors
    ]])

    # ⚡ Predict
    prediction = model_2.predict(input_array)
    print("Model Prediction for non English speaker: ",prediction)

    return {"prediction": int(prediction[0])}