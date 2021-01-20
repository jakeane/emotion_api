from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import Dict

from emotion_model import ApiModel, get_model

app = FastAPI(name="GoEmotion Model", description="Multi-label emotion model built on top of BERT")


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    probabilities: Dict[str, float]

@app.get("/")
def root_message():
    return {"message": "Use /predict with \"text\" field to get emotions"}

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest, model: ApiModel = Depends(get_model)):
    sorted_probabilities = model.predict(request.text)
    return SentimentResponse(
        probabilities=sorted_probabilities
    )
