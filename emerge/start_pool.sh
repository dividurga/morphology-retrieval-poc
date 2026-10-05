#!/bin/sh
# N headless coppeliasim instances, zmq ports 23200 .. 23200+N-1 (set BASE to change), logs emerge/cs_<port>.log. each needs an open stdin.
N=${1:-8}
cd "$(dirname "$0")/vendor"
for k in $(seq 0 $((N - 1))); do
  P=$((${BASE:-23200} + k))
  W=$((${BASE:-23200} + 500 + k))
  nohup sh -c "tail -f /dev/null | ./coppeliaSim.app/Contents/MacOS/coppeliaSim -H -GzmqRemoteApi.rpcPort=$P -GwsRemoteApi.port=$W > ../cs_$P.log 2>&1" > /dev/null 2>&1 &
  sleep 3
done
