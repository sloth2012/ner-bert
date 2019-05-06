#!/usr/bin/env bash

container_id=`docker ps -a -f ancestor=ner-bert -q`
if [ ! -z "${container_id}" -a "${container_id}" != " " ]; then
    echo "restart ner-bert"
    docker restart ${container_id}
else
    echo "create ner-bert"
    docker run -d --shm-size=300m -p 50002:50002 --memory=500m --restart=always ner-bert
fi