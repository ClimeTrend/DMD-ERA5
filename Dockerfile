# This Dockerfile is used to build a Docker image that will be used to run the tests marked with the docker marker.
FROM python:3.10
RUN apt-get update && apt-get install -y git
WORKDIR /app
COPY . /app
RUN git init
RUN pip install ".[dev]"
RUN dvc init
RUN dvc config core.autostage true
CMD ["sh", "-c", "pytest -s -m docker"]
