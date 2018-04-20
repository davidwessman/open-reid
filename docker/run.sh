#!/bin/bash
cd $(dirname $(dirname $0))
dir=$(pwd)
user_id=$(id -u $USER)
nvidia-docker run --shm-size 2048M --rm -e http_proxy=$http_proxy -e https_proxy=$http_proxy -it -v $dir:/home/user/open-reid syn_openreid_$user_id bash
