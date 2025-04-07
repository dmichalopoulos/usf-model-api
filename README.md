# usf-model-api

## Overview
This repo contains a prototype REST API service for serving model predictions from the `Store Item Demand
Forecasting Challenge` Kaggle competition. The web app is built using FastAPI, and is designed to 
be run as a containerized application using Docker. During image build, all `python` packaging and 
dependency management is handled by `pipenv`. See `/Dockerfile` for more details.

Once running, the app supports the option of generating predictions from one of two models, depending
on the contents of the `POST` request payload:
 * A `CatBoostRegressor`
 * A `LightGBMRegressor`

The models are relatively simple in nature, with a target variable called `sales`, and three (3) input 
features:
 * `date`
 * `store`
 * `item`

Note that I spent very little time trying to optimize (hyperparameter-tune) each of the models, and
instead focused on getting a well-functioning and easy-to-use app up and running. Future work would
include spending more time on the models themselves, including:
 * Cross-validated hyperparameter tuning
 * Feature engineering
  * Taking more advantage of time/seasonal patterns
  * Bringing in 3rd party datasets (e.g., seasonal data such as historical weather patterns by time 
  * of year, to test whether they affect sales)
 * Model selection (additional models, ensembling, stacking, etc.)
 * Model evaluation
   * For simplicity, I trained and evaluated the models using a simple randomized `train-test-split`
     approach. But since the models are time-series in nature, and the goal is to accurately forecast
     the future, a more appropriate approach would be to leverage time-based splits (including during
     any hyperparameter tuning and cross-validation). So please note that I am completely aware that
     a random `train-test` split isn't really appropriate here, but I wanted to focus the majority of
     my efforts on creating a clean, well-functioning, and easy-to-use application.

Upon launch, the app port-forwards to `0.0.0.0:80` on the host machine, and exposes the following 
endpoints:
 
 * `[GET] /` - Application root
 * `[GET] /sales-forecast` - Sales Forecasting router application root
 * `[POST] /sales-forecast/predict` - accepts a JSON payload containing the features for a single prediction, and returns the predicted value
 * `[GET] /sales-forecast/status` - Returns a 200 status code if the app is running

## Repository Design
This repo is organized into 4 main directories:
 * `/src`: Houses the main Python package (`usf-model-api`), which contains base class and utilities 
   for developing web services (using FastAPI), and developing ML models. Right now it's pretty lean, 
   and we only have one modeling application to speak of here. But my goal in designing it this way
   is to attempt to be forward-thinking towards a future where there this would serve as a library
   of common components used in the development of numerous model and service deployments.
 * `/models`: Application-specific directories (only 1 at the moment) containing code for training and 
   deploying ML models.
 * `/service`: Application-specific directories (only 1 at the moment) containing code for defining
   and deploying REST-based web services.
 * `/tests`: Unit tests for `/src/usf_model_api`. Note that there are also unit tests for the Sales
   Forecasting web service in `/service/routers/sales_forecasting/test_router.py`

In real life, `/src/usf_model_api`, `/models`, and `/service` would likely be completely separate repos
(and each with their own dedicated unit and integration tests). But for the purposes of this exercise, 
they are all packaged together here.

## Getting Started
> **NOTE**  
> You may need to run `chmod +x` on `build.sh` and `run.sh` to make them executable.

### Host Machine Prerequisites
 * Make sure you have `docker` installed and running
 * Your machine should be compatible with either `linux/amd64` or `linux/arm64` architectures

If you want to build and launch the app and start using it in one go, follow the directions in the 
[Quickstart](#quickstart) section below. Otherwise, if you want to follow the step-by-step instructions,
you can skip ahead to the [Step-by-Step Instructions](#step-by-step-instructions) section.

### Quickstart
After you've cloned this repository, create a `./downloads` directory at the root level, and place 
your `train.csv` in there. Make sure you have `docker` installed and running. Then run the following:

```shell
cd usf-model-api  # If you're not already in the repo root
./build.sh [<|arm|amd>]  # Leave blank or choose `arm` for M1 Macs, `amd` for Intel machines
./run.sh launch  # Trains the models, and launches the web app
```

If successful, the app will be port-forwarded to `0.0.0.0:80`, and your setup is complete. For 
directions on how to use and interact with the app and its API, see [Using the Web App](#using-the-web-app).

### Step-by-Step Instructions
If you've skipped the [Quickstart](#quickstart) because you prefer a step-by-step setup experience,
you've come to the right place.

#### Build the Docker Image
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
> 
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

#### Launching the Application
Once the image is built, getting the app up and running can be fully managed `./run.sh` located in
the repo root. The training dataset is the `train.csv` file associated with the Kaggle competition, 
which can be downloaded [here](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data>).


> **IMPORTANT**  
> 
> Make sure that `/downloads/train.csv` exists.


> **TIPS**  
> 
> In what follows, Steps 2 (model training) and 3 (app launch) be run individually. But you can also
you can also execute both steps in one go by running:
> ```shell
> ./run.sh launch
> ````

If you go this route, you can skip Steps 2 and 3 below and go straight to [Using the Web App](#using-the-web-app).

#### (1) Running Unit Tests [Optional]
Several unit tests were created during app development, and they are included if you wish to run them:
```shell
./run.sh pytest
```

#### (2) Training the Models
> **NOTE**  
> 
> If you already ran `./run.sh launch`, you can skip ahead to [Using the Web App](#using-the-web-app).

Before the web app can be launched, the `catboost` and `lightgbm` models must first be trained and
saved locally (serialized as `.pkl` files). Doing so is as simple as running
```shell
./run.sh train
```
After running the `train` command, the serialized models will be saved in the `/service/routers/sales_forecasting/assets`
directory of the `usf-model-api-root` volume that was created during the image build process.


> **WARNING**  
> 
> Running `./run.sh train` always overwrite any existing saved models in `/service/routers/sales_forecasting/assets`.
> A simple fix for this would have been to append timestamps to the saved model filenames, but for simplicity
> I have neglected to do so.


#### (3) Launching the Web App
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

Now the webapp is up and running, and should be port-forwarding to `0.0.0.0:80` on the host machine.
In the next section ([Using the Web App](#using-the-web-app)), we provide instructions and examples 
on how to start using the app, and interacting with its API.

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
Note that there is `pydantic` validation performed on the `date` field, so make sure each prediction
request has this value formatted as `yyyy-MM-dd`. If you submit a request where any `date` field values
are malformed, you will get back an error message like the following:

Request:
```shell
# 2nd object has malformed date '2025-5-01'
curl -X POST -H "Content-Type: application/json" -d \
  '[{"model_id": "catboost", "date": "2025-04-01", "store": 1, "item": 2}, {"model_id": "lgbm", "date": "2025-5-01", "store": 1, "item": 2}]' \
  http://0.0.0.0:80/sales-forecasting/predict
```

Response:
```json
{
    "detail": [
        {
            "type": "model_attributes_type",
            "loc": [
                "body",
                "SalesForecastRequest"
            ],
            "msg": "Input should be a valid dictionary or object to extract fields from",
            "input": [
                {
                    "model_id": "catboost",
                    "date": "2025-04-01",
                    "store": 1,
                    "item": 2
                },
                {
                    "model_id": "lgbm",
                    "date": "2025-5-01",
                    "store": 1,
                    "item": 2
                }
            ]
        },
        {
            "type": "value_error",
            "loc": [
                "body",
                "list[SalesForecastRequest]",
                1,
                "date"
            ],
            "msg": "Value error, Invalid date format '2025-5-01'. Expected format 'yyyy-MM-dd'.",
            "input": "2025-5-01",
            "ctx": {
                "error": {}
            }
        }
    ]
}
```
