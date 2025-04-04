docker volume rm usf-model-api-root
docker build --no-cache -t usf-model-api:latest .
docker volume create usf-model-api-root
