#!/bin/sh
# kill any coppeliasim (and its stdin holder) and start a fresh headless one. use it when the zmq api stops answering.
pkill -9 -f "coppeliaSim.app"; pkill -9 -f "tail -f /dev/null"; sleep 3
sh "$(dirname "$0")/start_coppelia.sh"
sleep 25
