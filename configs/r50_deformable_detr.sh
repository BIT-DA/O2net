#!/usr/bin/env bash

set -x

EXP_DIR=$0
PY_ARGS=${@:1}

python -u main.py \
    ${PY_ARGS}

#
