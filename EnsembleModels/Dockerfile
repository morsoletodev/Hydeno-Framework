FROM quay.io/jupyter/scipy-notebook
WORKDIR /home/jovyan/work

COPY pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir .
