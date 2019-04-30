FROM frolvlad/alpine-miniconda3:python3.6

LABEL maintainer="lx<liuxiang@bailian.ai>"

ENV LIBRARY_PATH=/lib:/usr/lib
ENV WORKER_NUM=1
WORKDIR /pos-bert
COPY packages pkgs
COPY docker/run.py run.py
COPY requirements.txt requirements.txt

RUN apk add --no-cache gcc \
    g++ \
    && pip install --upgrade pip \
    && pip3 install -r requirements.txt --no-index --find-links file:///pos-bert/pkgs \
    && pip install pkgs/bailian_nlp*.tar.gz \
    && rm -rf pkgs

EXPOSE 50001
CMD ["python", "run.py", "workers=$WORKER_NUM"]


