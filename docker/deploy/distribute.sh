#!/usr/bin/env bash

pos_machines=(
    39.105.135.43
#    60.205.204.174
#    47.93.206.155
#    39.105.102.226
#    39.105.87.141
)

ner_machines=(
    47.94.131.238
#    39.105.135.46
#    59.110.218.101
#    39.107.98.198
#    47.93.59.46
)

current_dir=$(dirname $0)
docker_dir=$(dirname ${current_dir})

image_type=$1
image_name=${image_type}-bert
image_file_name=${image_name}.tar.gz
image_file=${docker_dir}/${image_file_name}

eval machines=(\"\${${image_type}_machines[@]}\")

run_script_file=${docker_dir}/${image_type}/run.sh
load_script_file=${docker_dir}/${image_type}/load.sh
deploy_script_file=${current_dir}/deploy.sh
rsa_file=${current_dir}/bailian

for ip in "${machines[@]}";do
    echo "rsync ${image_file} to $ip"
    port=11000
    user="liuxiang"
    target_dir=/tmp
    rsync -qav --progress -e "ssh -p ${port} -i ${rsa_file} -o StrictHostKeyChecking=no" ${deploy_script_file} ${image_file} ${load_script_file} ${run_script_file} ${user}@${ip}:${target_dir}
    ssh -p ${port} -i ${rsa_file} -o StrictHostKeyChecking=no ${user}@${ip} "sudo su - root -c '${target_dir}/deploy.sh ${target_dir}/${image_file_name}'"
done