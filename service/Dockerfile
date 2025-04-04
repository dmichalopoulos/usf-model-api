FROM usf-model-api:latest

EXPOSE 80
WORKDIR /package/service
CMD ["pipenv", "run", "fastapi", "run", "./app.py", "--port", "80"]

#WORKDIR /agm/budget_optimization
#
#COPY dashboard.py dashboard.py
#COPY dashboard_common_components.py dashboard_common_components.py
#COPY dashboard_predictor_components.py dashboard_predictor_components.py
#COPY dashboard_optimizer_components.py dashboard_optimizer_components.py
#COPY assets assets

#EXPOSE 80

#RUN pip install gradio humanize pandas requests
#RUN pip install atg_model_serving
#RUN pip install --no-build-isolation seldon-core==1.15.0
#
#CMD python dashboard.py