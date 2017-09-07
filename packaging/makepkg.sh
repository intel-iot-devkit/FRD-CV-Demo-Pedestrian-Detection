#!/usr/bin/env bash
# Generate a package file
# Takes the CMake source path as input
TEMP_DIR="$(mktemp -d)"
mkdir $TEMP_DIR/pddemo
cp pddemo lib*.so $TEMP_DIR/pddemo

LIBRARIES="$(ldd pddemo | sed -ne 's/^.*=> //p' | sed -e 's/ (0x[0-9a-f]*)$//' | grep -v 'libc\.so')"
mkdir $TEMP_DIR/pddemo/libs
cp $LIBRARIES $TEMP_DIR/pddemo/libs
cp $1/packaging/run.sh $TEMP_DIR/pddemo/

tar -C $TEMP_DIR -czvf pddemo.tar.gz pddemo
rm -r $TEMP_DIR
