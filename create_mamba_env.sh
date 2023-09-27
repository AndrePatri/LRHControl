#!/bin/bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# should match env name from YAML
ENV_NAME=lrhc_examples

pushd "${ROOT_DIR}/"

    # setup mamba
    MAMBA_DIR="$(mamba info --base)"
    
    source "${MAMBA_DIR}/etc/profile.d/mamba.sh"

    # !!! this removes existing version of the env
    mamba remove -y -n "${ENV_NAME}" --all

    # create the env from YAML
    mamba env create -f ./lrhc_examples_mamba_env.yml

    # activate env
    # mamba activate "${ENV_NAME}"

    # # install lrhc_examples package in editable mode
    # pip install -e .

popd

echo "SUCCESS"