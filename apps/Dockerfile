FROM usf-model-api:latest

#ENV HOME /root
#ADD .vimrc ${HOME}

## Install gcloud CLI
#RUN export HOME=/usr/local && curl -sSL https://sdk.cloud.google.com | bash
#ENV PATH "${PATH}":/usr/local/google-cloud-sdk/bin




#FROM us.gcr.io/altway-internal/awareness-growth-mkt:latest

#WORKDIR /agm/budget_optimization
#
#COPY dashboard.py dashboard.py
#COPY dashboard_common_components.py dashboard_common_components.py
#COPY dashboard_predictor_components.py dashboard_predictor_components.py
#COPY dashboard_optimizer_components.py dashboard_optimizer_components.py
#COPY assets assets

EXPOSE 80

RUN pip install gradio humanize pandas requests
RUN pip install atg_model_serving
RUN pip install --no-build-isolation seldon-core==1.15.0

CMD python dashboard.py






ARG baseImage=us.gcr.io/altway-internal/pipenv-base-py31013:latest
FROM ${baseImage}

ARG gitsha=unknown
ENV GIT_SHA=${gitsha}
ENV GIT_REPO=atg-research

COPY . package/
WORKDIR /package

RUN --mount=type=secret,id=gcloud-creds,dst=/root/.config/gcloud/application_default_credentials.json pipenv sync --clear

CMD /bin/bash