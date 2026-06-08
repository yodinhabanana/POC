from fastapi import FastAPI
import pandas as pd

app = FastAPI()

# joblib may not be installed in all environments; fall back to pickle
try:
    import joblib

    def load_model(path: str):
        return joblib.load(path)
except Exception:
    import pickle

    def load_model(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

model = load_model("student_model.pkl")


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"high_performance": int(prediction[0])}