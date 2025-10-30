from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load model
with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

app = FastAPI()

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(client: Client):
    probability = pipeline.predict_proba([client.dict()])[0][1]
    return {"conversion_probability": round(probability, 3)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
