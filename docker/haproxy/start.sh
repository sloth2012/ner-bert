#!/usr/bin/env bash

docker run -d --name haproxy -v /Users/lx/PycharmProjects/pos-bert/docker/haproxy:/usr/local/etc/haproxy:ro haproxy
