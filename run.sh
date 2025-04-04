#!/bin/bash

set -ex

COMMAND="$1"

if [[ train == $COMMAND ]]; then
    docker run -v usf-model-api-root:/package usf-model-api:latest pipenv run python ./models/sales_forecasting/train.py \
      --model-name "catboost" \
      --model-name "lgbm"
elif [[ serve == $COMMAND ]]; then
    docker run -v usf-model-api-root:/package -p 80:80 usf-model-api:latest pipenv run fastapi run ./service/app.py --port 80
else
    echo "No command provided. Available commands: train, serve"
fi
