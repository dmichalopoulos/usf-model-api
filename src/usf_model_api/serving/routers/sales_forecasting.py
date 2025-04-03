from http import HTTPStatus
from logging import getLogger
from dateutil.parser import parse

from pydantic import field_validator
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import Response, JSONResponse

from usf_model_api.serving.base import PredictionRequest


router = APIRouter(
    prefix="/sales-forecasting",
    tags=["ai_model"],
    dependencies=None,
    # responses={404: {"description": "Not found"}},
)
db = {}


LOG = getLogger(__name__)


class SalesForecastRequest(PredictionRequest):
    date: str
    store_id: int
    item_id: int

    @field_validator("date")
    def check_date(cls, date: str):
        try:
            parse(date)
        except ValueError:
            LOG.exception(
                "Invalid date value '%s'. Ensure you pass in a valid date with format 'yyyy-MM-dd'",
                date,
            )
            raise

        return date


@router.get("/")
def read_root():
    return {"message": "Welcome to the Sales Forecasting AI Model service!"}


@router.get("/status", status_code=HTTPStatus.OK)
def get_app_status():
    return {"message": "The app is up and running."}


@router.post("/predict")
def predict(prediction_request: SalesForecastRequest):
    assert isinstance(prediction_request, SalesForecastRequest), "Not a SalesForecastRequest"
    return 1


# @router.get("/list", response_class=JSONResponse)
# def list_all_foo() -> dict:
#     return db
#
#
# @router.get("/{a}", response_class=Response)
# def read_foo(a: int):
#     foo = db.get(a)
#     if foo is None:
#         raise HTTPException(status_code=404, detail=f"Object with a={a} does not exist.")
#
#     return Response(pickle.dumps(foo), media_type="binary/octet-stream")
#
#
# @router.post("/{a}", response_class=JSONResponse)
# def create_foo(a: int):
#     crud_type = "create" if a not in db else "update"
#     db[a] = Foo(a=a)
#
#     return JSONResponse(
#         status_code=200,
#         content={
#             "message": f"Foo (a={a}) successfully created/updated.",
#             "crud_type": crud_type,
#         },
#     )
