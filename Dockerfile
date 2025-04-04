FROM python:3.10.13-slim-bullseye

RUN apt-get update && apt-get -y upgrade

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

RUN pip3 install pipenv==2024.4.1 pip

COPY . package/
WORKDIR /package

RUN pipenv sync --clear
CMD /bin/bash
