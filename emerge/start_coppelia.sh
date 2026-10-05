#!/bin/sh
# headless coppeliasim, kept alive by an open stdin (it quits on stdin EOF). zmq api on port 23000.
cd "$(dirname "$0")/vendor"
nohup sh -c 'tail -f /dev/null | ./coppeliaSim.app/Contents/MacOS/coppeliaSim -H > ../cs.log 2>&1' > /dev/null 2>&1 &
