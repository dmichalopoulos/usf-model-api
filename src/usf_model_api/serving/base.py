from pydantic import BaseModel


class PredictionRequest(BaseModel):
    model_id: str
