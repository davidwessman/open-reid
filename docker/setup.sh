#!/bin/bash
cd $(dirname $0)
user_id=$(id -u $USER)
group_id=$(id -g $USER)
docker build --build-arg http_proxy="$http_proxy" --build-arg user_id=$user_id --build-arg group_id=$group_id -t syn_openreid_$user_id .
