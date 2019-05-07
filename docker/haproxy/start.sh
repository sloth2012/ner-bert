#!/usr/bin/env bash

current_dir=$(dirname $0)

container_id=`docker ps -a -f ancestor=haproxy -q`
if [ ! -z "${container_id}" -a "${container_id}" != " " ]; then
    echo "stop and remove haproxy containers..."
    docker stop ${container_id} && docker rm ${container_id}
fi
docker run -d --name haproxy -p 50001:50001 -p 50002:50002 -p 50000:50000 -v ${current_dir}:/usr/local/etc/haproxy:ro haproxy
