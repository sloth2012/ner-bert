#!/usr/bin/env bash

container_id=`docker ps -a -f ancestor=pos-bert -q`
if [ ! -z "${container_id}" -a "${container_id}" != " " ]; then
    echo "restart pos-bert"
    docker restart ${container_id}
else
    echo "create pos-bert"
    docker run -d --shm-size=300m -p 50001:50001 --memory=500m --restart=always pos-bert
fi