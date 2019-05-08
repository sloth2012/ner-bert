#!/usr/bin/env bash

image_type="pos"
image_name=${image_type}-bert
container_id=`docker ps -a -f ancestor=${image_name} -q`
if [ ! -z "${container_id}" -a "${container_id}" != " " ]; then
    echo "stop and remove ${image_type}-bert containers..."
    docker stop ${container_id} && docker rm ${container_id}
fi

image_id=`docker images ${image_name} -q`

if [ ! -z "${image_id}" -a "${image_id}" != " " ]; then
    echo "remove ${image_type}-bert image..."
    docker rmi ${image_id}
fi

if [ "x" != "x"$1 ]; then
    # 说明是传入了文件参数
    image_file=$1
else
    current_dir=$(dirname $0)
    docker_dir=$(dirname ${current_dir})
    image_file=${docker_dir}/${image_name}.tar.gz
fi

gunzip -c ${image_file} | docker load