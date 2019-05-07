#!/usr/bin/env bash

haproxy_dir=$(dirname $0)
docker_dir=$(dirname ${haproxy_dir})

ip=39.96.167.90
port=11000
rsa_file=${docker_dir}/deploy/bailian
user=liuxiang
target_dir=/home/${user}/projects
rsync -qav --progress -e "ssh -p ${port} -i ${rsa_file} -o StrictHostKeyChecking=no" ${haproxy_dir} ${user}@${ip}:${target_dir}
ssh -p ${port} -i ${rsa_file} -o StrictHostKeyChecking=no ${user}@${ip} "sudo su - root -c '${target_dir}/haproxy/start.sh'"


