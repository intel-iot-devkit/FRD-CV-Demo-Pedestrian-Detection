#!/usr/bin/env bash
ORIG_CWD="$(pwd)"
SCRIPT_PATH="$(dirname "$0")"
SCRIPT_PATH="$(realpath $SCRIPT_PATH)"
cd $ORIG_CWD

LD_LIBRARY_PATH="$SCRIPT_PATH/libs" $SCRIPT_PATH/pddemo $@
