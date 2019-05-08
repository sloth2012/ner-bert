#!/usr/bin/env bash

current_dir=$(cd `dirname $0`; pwd)
docker_dir=$(dirname ${current_dir})

docker save pos-bert | gzip > ${docker_dir}/pos-bert.tar.gz