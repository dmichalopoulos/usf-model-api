#!/bin/bash

set -ex

COMMAND="$1"

run_train() {
  docker run -v usf-model-api-root:/package usf-model-api:latest pipenv run python ./models/sales_forecasting/train.py \
      --model-name "catboost" \
      --model-name "lgbm"
}

run_serve() {
  docker run -v usf-model-api-root:/package -p 80:80 usf-model-api:latest pipenv run fastapi run ./service/api.py --port 80
}

run_pytest() {
  docker run -v usf-model-api-root:/package usf-model-api:latest pipenv run pytest ./tests ./service/routers/sales_forecasting/test_router.py
}

if [[ train == $COMMAND ]]; then
    run_train
elif [[ serve == $COMMAND ]]; then
    run_serve
elif [[ launch == $COMMAND ]]; then
    run_train && wait && run_serve
elif [[ pytest == $COMMAND ]]; then
    docker run -v usf-model-api-root:/package usf-model-api:latest pipenv run pytest ./tests ./service/routers/sales_forecasting/test_router.py
else
    echo "No command provided. Available commands: train, serve, launch, and pytest"
fi
