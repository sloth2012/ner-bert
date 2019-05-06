#!/usr/bin/env bash

docker run -d --name haproxy -p 8123:50000 -p 8124:50001 -v /Users/lx/PycharmProjects/pos-bert/docker/haproxy:/usr/local/etc/haproxy:ro haproxy
