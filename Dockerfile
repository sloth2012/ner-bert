FROM frolvlad/alpine-miniconda3:python3.6 as builder

WORKDIR /pos-bert

ENV WORKER_NUM=1
ENV LIBRARY_PATH=/lib:/usr/lib
ENV PYTHONPATH=/pos-bert

COPY pkgs pkgs
COPY bailian_nlp bailian_nlp
COPY requirements.txt requirements.txt

RUN apk add --no-cache gcc \
    g++ \
    jpeg-dev \
    && pip install --upgrade pip \
#    && pip install --no-cache-dir -r requirements.txt \
    && pip3 install -r requirements.txt --no-cache-dir --no-index --find-links file:///pos-bert/pkgs \
    && rm -rf pkgs \
    && rm -rf requirements.txt


FROM scratch
LABEL maintainer="lx<liuxiang@bailian.ai>"

COPY --from=builder / /
WORKDIR /pos-bert

ENV WORKER_NUM=1
ENV LIBRARY_PATH=/lib:/usr/lib
ENV PYTHONPATH=/pos-bert:$PYTHONPATH
ENV CONDA_DIR="/opt/conda"
ENV PATH="$CONDA_DIR/bin:$PATH"

EXPOSE 50001
CMD ["python", "bailian_nlp/web/server.py", "workers=$WORKER_NUM"]


