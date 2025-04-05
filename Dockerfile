# Python base image
FROM python:3.10.13-slim-bullseye

# Install a few supporting build tools
RUN apt-get update && apt-get -y upgrade
RUN apt-get -y install curl && apt-get install libgomp1  # for LightGBM

# Set up and build python environment w/ pipenv
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

RUN pip3 install pipenv==2024.4.1 pip

COPY . package/
WORKDIR /package

RUN pipenv lock && pipenv sync --clear

# Expose for FastAPI service + port-forwarding
EXPOSE 80

CMD /bin/bash
