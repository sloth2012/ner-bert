#!/usr/bin/env bash

image_type="pos"
image_name=${image_type}-bert
container_id=`docker ps -a -f ancestor=${image_name} -q`
if [ ! -z "${container_id}" -a "${container_id}" != " " ]; then
    echo "restart ${image_name}"
    docker restart ${container_id}
else
    echo "create ${image_name}"
    docker run -d --shm-size=150m -p 50009:50001 --memory=500m --restart=always ${image_name}
fi