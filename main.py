from fastapi import FastAPI
from pydantic import BaseModel

import joblib
model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

class InputFeatures(BaseModel):
    minutes_played:int
    highest_value: int

def preprocessing(input_features: InputFeatures):
    dict_f = {
        'minutes_played': input_features.minutes_played,
        'highest_value': input_features.highest_value,
        
    }
    features_list = [dict_f[key] for key in sorted(dict_f)]
    scaled_features = scaler.transform([list(dict_f.values())])
    return scaled_features.tolist()

@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]} 
