# usf-model-api

## Summary
This repo contains a prototype REST API service for serving model predictions from the `Store Item Demand
Forecasting Challenge` Kaggle competition. The web app is built using FastAPI, and is designed to 
be run as a containerized application using Docker. During image build, all `python` packaging and 
dependency management is handled by `pipenv`. See `./Dockerfile` for more details.

Once running, the app supports the option of generating predictions from one of two models, depending
on the contents of the `POST` request payload:
 * A `CatBoostRegressor`
 * A `LightGBMRegressor`

Upon launch, the app port-forwards to 0.0.0.0:80 on the host machine, and exposes the following 
endpoints:
 
 * `[GET] /sales-forecast` - Application root
 * `[POST] /sales-forecast/predict` - accepts a JSON payload containing the features for a single prediction, and returns the predicted value
 * `[GET] /sales-forecast/status` - Returns a 200 status code if the app is running

## Getting Started
### Host Machine Prerequisites
 * `docker` installed and running
 * Compatible with either `linux/amd64` or `linux/arm64` architectures

### Build the Docker Image
From the root of the repo:

 * To build the `linux/arm64`-compatible image, run:
   ```shell
   ./build.sh arm 
   ```
   Note that `arm` is the default architecture, so you can also just run `./build.sh` (no args) to 
   build the `arm64` image.


 * To build the `linux/amd64`-compatible image, run:
   ```shell
   ./build.sh amd
   ```
> **WARNING**  
> * If your host machine is an M1 Mac, building the `arm64` image is recommended. It will build much
   faster, and likely run more efficiently than the `amd64` image.
> * Conversely, if your host machine is an Intel Mac, building the `amd64` image is recommended.
> * In either case, if you have an architecture mismatch between the host and the image, you may see
   messages like the following when you run the steps in the next section:
> ```shell
>      WARNING: The requested image\'s platform (linux/amd64) does not match the detected host platform 
>      (linux/arm64/v8) and no specific platform was requested
> ```
> This is expected, and can be ignored.

## Launching the Application
Once the image is built, getting the app up and running can be fully managed `./run.sh` located in
the repo root.

> **TIP**  
> Steps (2) and (3) below can be run sequentially, but if you want to combine them into a single command,
you can run:
> ```shell
> ./run.sh launch
> ````

### (1) Running Unit Tests (Optional)
Several unit tests were created during app development, and they are included if you wish to run them:
```shell
./run.sh pytest
```

### (2) Training the Models
> **NOTE**  
> If you already ran `./run.sh launch`, you can skip this step, as the models will already be trained and saved.

Before the web app can be launched, the `catboost` and `lightgbm` models must first be trained and
saved locally (serialized as `.pkl` files). Doing so is another simple one-liner:
```shell
./run.sh train
```
After running the `train` command, the serialized models will be saved in the `/service/routers/sales_forecasting/assets`
directory of the `usf-model-api-root` volume that was created during the image build process.

> **WARNING**
> * Running `train` before running the `build` command will result in an error, as the 
   saved model artifacts will not exist yet. 
> * Running `train` always overwrites any existing saved models in `/service/routers/sales_forecasting/assets`.

### (3) Launching the Web App
> **NOTE**  
> If you already ran `./run.sh launch`, you can skip this step, since the web app should already be
> up and running.

Finally, we are ready to launch the web app:
```shell
./run.sh serve
```

This will run the app in the foreground on the `FastAPI` server, which will display output that looks
like the following:
```shell
FastAPI      Starting production server 🚀
 
             Searching for package file structure from directories with         
             __init__.py files                                                  
INFO:usf_model_api.serving.utils:Loading saved model file '/package/service/routers/sales_forecasting/assets/lgbm.pkl'
INFO:usf_model_api.serving.utils:Loading saved model file '/package/service/routers/sales_forecasting/assets/catboost.pkl'
             Importing from /package
 
    module   📁 service        
             ├── 🐍 __init__.py
             └── 🐍 api.py     
 
      code   Importing the FastAPI app object from the module with the following
             code:                                                              
 
             from service.api import app
 
       app   Using import string: service.api:app
 
    server   Server started at http://0.0.0.0:80
    server   Documentation at http://0.0.0.0:80/docs
 
             Logs:
 
      INFO   Started server process [1]
INFO:uvicorn.error:Started server process [1]
      INFO   Waiting for application startup.
INFO:uvicorn.error:Waiting for application startup.
      INFO   Application startup complete.
INFO:uvicorn.error:Application startup complete.
      INFO   Uvicorn running on http://0.0.0.0:80 (Press CTRL+C to quit)
INFO:uvicorn.error:Uvicorn running on http://0.0.0.0:80 (Press CTRL+C to quit)
```

Now the webapp is up and running, and additionally is port-forwarding to `0.0.0.0:80` on the host 
machine.

## Using the Web App
### Viewing the API Docs
You can view the Swagger documentation automatically generated by `FastAPI` by navigating to `http://0.0.0.0:80/docs`
in your browser.

### Checking The Status (`[GET] /sales-forecasting/status`)
To confirm that the app is running, you can send a request to this endpoint, like so:

Request:
```shell
curl -X GET -H "Content-Type: application/json" http://0.0.0.0:80/sales-forecasting/status
````

Response:
```json
{
    "message": "Status Code 200: The app is up and running."
}
```

### Sending Prediction Requests (`[POST] /sales-forecasting/predict`)
The web app will automatically route prediction requests to either the `catboost` or `lightgbm` model, 
depending on the `model_id` field value in each `SalesPredictionRequest` object in the `POST` request 
payload. For example:

Request:
```shell
curl -X POST -H "Content-Type: application/json" -d \
  '[{"model_id": "catboost", "date": "2025-04-01", "store": 1, "item": 2}, {"model_id": "lgbm", "date": "2025-04-01", "store": 1, "item": 2}, {"model_id": "catboost", "date": "2025-04-02", "store": 2, "item": 2}]' \
  http://0.0.0.0:80/sales-forecasting/predict
```

Response:
```json
{
    "message": "Prediction request successful.",
    "predictions": [
        {
            "prediction_id": "894068c7-225c-4c34-894d-c538976acbb0",
            "model_id": "catboost",
            "date": "2025-04-01",
            "store": 1,
            "item": 2,
            "prediction": 57.94393713166007,
            "created_at": "2025-04-05 00:52:24.118"
        },
        {
            "prediction_id": "27fbf974-b0f9-48d9-9d6c-e12a62f71af2",
            "model_id": "lgbm",
            "date": "2025-04-01",
            "store": 1,
            "item": 2,
            "prediction": 33.59795468522696,
            "created_at": "2025-04-05 00:52:24.161"
        },
        {
            "prediction_id": "2d6ff7e3-0631-46d7-98c0-c2267a32de74",
            "model_id": "catboost",
            "date": "2025-04-02",
            "store": 2,
            "item": 2,
            "prediction": 79.6860749891785,
            "created_at": "2025-04-05 00:52:24.118"
        }
    ]
}
```
