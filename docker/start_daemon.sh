#!/usr/bin/sh
container_name=ml_dev

docker kill ${container_name}

docker run -dt --rm \
  -v $(pwd):/home/data \
  -u $(id -u):$(id -g) \
  --name ${container_name} \
  ml_dev:b00001

ip=$(docker inspect -f "{{ .NetworkSettings.IPAddress }}" ${container_name})
echo "ssh tf-docker@${ip}"
