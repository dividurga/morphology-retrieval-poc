#!/bin/sh
# stop the instances the clean way: kill the stdin feeders, coppeliasim quits on stdin EOF.
# do NOT pkill the coppeliaSim processes: killed during start-up or exit they stay in state UE and hold their ports until reboot.
pkill -f "tail -f /dev/null"
