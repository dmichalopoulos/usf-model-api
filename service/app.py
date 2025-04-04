from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from service.routers import sales_forecasting


app = FastAPI()
# app = FastAPI(dependencies=[Depends(get_query_token)])
app.include_router(sales_forecasting.router)
# app.include_router(
#     admin.router,
#     prefix="/admin",
#     tags=["admin"],
#     dependencies=[Depends(get_token_header)],
#     responses={418: {"description": "I'm a teapot"}},
# )

"""
The @app.exception_handler(Exception) decorator in FastAPI is used to define a global exception 
handler for the application. This handler will catch all exceptions that are not explicitly handled 
by other parts of the application. In the provided code, the global_exception_handler function is 
defined to handle these exceptions. When an unhandled exception occurs, this function will return a
JSON response with a status code of 500 and a message indicating that an unexpected error occurred.
"""


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
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
    return JSONResponse(status_code=200, content={"message": "Welcome to my FastAPI service"})
