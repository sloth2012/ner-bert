#!/usr/bin/env bash

current_dir=$(dirname $0)
${current_dir}/load.sh $1 \
&& ${current_dir}/run.sh \
&& echo "done the load and run!"

