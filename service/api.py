from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from service.routers.sales_forecasting.router import router


app = FastAPI()
app.include_router(router)
# In the future, we can add more routers like this:
# app.include_router(
#     something.router,
#     prefix="/something",
#     tags=["this_is_something"],
#     dependencies=[Depends(get_token_header)],
#     responses={418: {"description": "I'm a teapot"}},
# )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    The @app.exception_handler(Exception) decorator in FastAPI is used to define a global exception
    handler for the application. This handler will catch all exceptions that are not explicitly handled
    by other parts of the application. In the provided code, the global_exception_handler function is
    defined to handle these exceptions. When an unhandled exception occurs, this function will return a
    JSON response with a status code of 500 and a message indicating that an unexpected error occurred.
    """
    return JSONResponse(
        status_code=500,
        content={
            "message": "An unexpected error occurred.",
            "request_url": str(request.url),
            "exception": str(exc),
        },
    )


@app.get("/", response_class=JSONResponse)
def read_root() -> JSONResponse:
    return JSONResponse(
        status_code=200, content={"message": "Welcome to my Model Prediction Service"}
    )
