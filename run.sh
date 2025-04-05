#!/bin/bash

set -ex

COMMAND="$1"

if [[ train == $COMMAND ]]; then
    docker run -v usf-model-api-root:/package usf-model-api:latest pipenv run python ./models/sales_forecasting/train.py \
      --model-name "catboost" \
      --model-name "lgbm"
elif [[ serve == $COMMAND ]]; then
    docker run -v usf-model-api-root:/package -p 80:80 usf-model-api:latest pipenv run fastapi run ./service/api.py --port 80
elif [[ pytest == $COMMAND ]]; then
    docker run -v usf-model-api-root:/package usf-model-api:latest pipenv run pytest ./tests ./service/routers/sales_forecasting/test_router.py
else
    echo "No command provided. Available commands: train, serve, and pytest"
fi
