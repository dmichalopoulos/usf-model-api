#set -ex

COMMAND="$1"

docker volume rm usf-model-api-root
#docker build --no-cache -t usf-model-api:latest .
#docker build --platform linux/amd64 --no-cache -t usf-model-api:latest .

# Build arm64 if specified, or by default
if [[ arm == $COMMAND ]] || [[ -z $COMMAND ]]; then
    docker build --platform linux/arm64 --no-cache -t usf-model-api:latest .
# Build amd64 if specified
elif [[ amd == $COMMAND ]]; then
    docker build --platform linux/amd64 --no-cache -t usf-model-api:latest .
else
    echo "Invalid command provided. Available commands: arm, amd."
fi

docker volume create usf-model-api-root
