#!/bin/bash
# usage: run_one.sh <args...>; log goes to logs/<name>.log
name=$(echo "$@" | sed -E 's/.*--name ([^ ]+).*/\1/')
python tools/run_experiment.py "$@" > "logs/$name.log" 2>&1
echo "done $name"
