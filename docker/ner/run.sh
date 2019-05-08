#!/usr/bin/env bash

image_type="ner"
image_name=${image_type}-bert
container_id=`docker ps -a -f ancestor=${image_name} -q`
if [ ! -z "${container_id}" -a "${container_id}" != " " ]; then
    echo "restart ${image_name}"
    docker restart ${container_id}
else
    echo "create ${image_name}"
    docker run -d --shm-size=300m -p 50010:50002 --memory=1000m --restart=always ${image_name}
fi