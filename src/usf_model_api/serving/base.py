from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import LogisticRegression

from usf_model_api.serving.utils import pickle_deserialize


class PredictionRequest(BaseModel):
    model_id: str


class SerdeBaseModel(BaseModel):
    @classmethod
    def from_bytes(cls, data) -> "SerdeBaseModel":
        return pickle_deserialize(cls, data)


class Item(SerdeBaseModel):
    item_id: int
    name: str
    price: float
    description: (str | None) = None


class Foo(SerdeBaseModel):
    a: int = 1
    b: Item = Item(item_id=1, name=f"Foo item", price=10, description="Foo class item")


class LogRegModel(SerdeBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    clf: LogisticRegression = LogisticRegression()
