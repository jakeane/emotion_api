from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List, Dict

from emotion_model import ApiModel, get_model

app = FastAPI(name="GoEmotion Model", description="Multi-label emotion model built on top of BERT")


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    emotion: Dict[str, float]
    animations: List[float]

@app.get("/")
def root_message():
    return {"message": "Use /predict with \"text\" field to get emotions"}

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest, model: ApiModel = Depends(get_model)):
    res = model.predict(request.text)
    return SentimentResponse(
        emotion=res['emotions'],
        animations=res['animations']
    )
