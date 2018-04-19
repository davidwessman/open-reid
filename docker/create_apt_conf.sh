#!/bin/bash

# Creates apt.conf if http_proxy is present
if [ "$http_proxy" != "" ]
then
  socks=$(echo $http_proxy | sed 's/http*/socks/')
  echo "Acquire::http::proxy \"$http_proxy\";" > /etc/apt/apt.conf
  echo "Acquire::https::proxy \"$http_proxy\";" >> /etc/apt/apt.conf
  echo "Acquire::socks::proxy \"$socks\";" >> /etc/apt/apt.conf
fi
