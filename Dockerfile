FROM continuumio/miniconda3:4.7.12

# replace dockers shell used by run to bash such that 'source activate' works
RUN ln -fs /bin/bash /bin/sh

RUN mkdir -p opt/rossmann
COPY requirements.txt opt/rossmann/
ENV PYTHONPATH=/home/rossmann

RUN conda create -n rossmann python=3.7 --yes

RUN source activate rossmann && \
    pip install -r opt/rossmann/requirements.txt \
    && source deactivate
