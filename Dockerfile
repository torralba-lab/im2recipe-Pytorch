FROM python:3.7.3-slim-stretch
RUN apt-get -y update && apt-get -y install gcc
RUN pip --no-cache-dir install --upgrade cython
WORKDIR /
COPY requirements.txt requirements.txt
RUN pip --no-cache-dir install -r requirements.txt
RUN apt-get -y install g++
RUN pip --no-cache-dir install torchwordemb
COPY scripts/ scripts/
COPY *.py /

ENTRYPOINT ["bash"]
