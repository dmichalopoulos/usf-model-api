FROM python:3.10.13-slim-bullseye

## Install a few supporting build tools
#RUN apt-get update && apt-get -y upgrade \
#  && apt-get install -y bc wget curl git zip unzip make vim jq netcat rsync screen procps \
#         build-essential libopenmpi-dev libpq-dev python-dev#\
         #default-libmysqlclient-dev default-mysql-client \

RUN apt-get update && apt-get -y upgrade
#build-essential libopenmpi-dev libpq-dev python-dev

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

#RUN pip3 install pipenv==2024.4.1 pip==22.3 setuptools==65.5.0 setuptools-scm==7.0.5 wheel==0.37.1
#RUN pip3 install pipenv==2024.4.1 pip==22.3 setuptools==65.5.0 setuptools-scm==7.0.5 wheel==0.37.1
RUN pip3 install pipenv==2024.4.1 pip #setuptools==78.1.0

COPY . package/
WORKDIR /package

#RUN --mount=type=secret,id=gcloud-creds,dst=/root/.config/gcloud/application_default_credentials.json pipenv sync --clear
RUN pipenv sync --clear
CMD /bin/bash
