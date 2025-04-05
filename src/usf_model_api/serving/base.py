from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """
    Prediction model request base class, which is subclassed by specific model request classes.

    Over time, more core attributes would be added to this, but for now, it simply requires the
    specification of a ``model_id``.

    Attributes
    ----------
    model_id : str
        The unique identifier for the model.
    """

    model_id: str
