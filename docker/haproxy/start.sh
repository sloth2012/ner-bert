#!/usr/bin/env bash

docker run -d -it --name haproxy -v /root/haproxy/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro haproxy
