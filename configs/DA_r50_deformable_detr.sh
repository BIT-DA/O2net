#!/usr/bin/env bash

set -x

EXP_DIR=$0
PY_ARGS=${@:1}

python -u DA_main.py \
    ${PY_ARGS}

#
