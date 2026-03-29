from fastapi import FastAPI
import pickle
import pandas as pd

# Load latest model
model = pickle.load(open("model_v2.pkl", "rb"))

app = FastAPI()

@app.get("/")
def home():
    return {"message": "ML API Running "}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    
    prediction = model.predict(df)
    
    return {"prediction": int(prediction[0])}