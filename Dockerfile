FROM python:3.7.3-slim-stretch
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -qq update && \
    apt-get -qq install gcc g++ && \
    apt-get --purge autoremove --yes && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
RUN pip --no-cache-dir install --upgrade cython && \
    pip --no-cache-dir install -r requirements.txt && \
    pip --no-cache-dir install torchwordemb
WORKDIR /
COPY scripts/ scripts/
COPY *.py /

ENTRYPOINT ["bash"]
