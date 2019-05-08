#!/usr/bin/env bash

current_dir=$(cd `dirname $0`; pwd)
project_dir=$(dirname $(dirname ${current_dir}))

cd ${project_dir} \
&& docker build -t pos-bert . && docker image prune -f \
