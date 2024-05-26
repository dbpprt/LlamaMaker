#!/bin/bash

if ! which conda >/dev/null 2>&1; then
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh -b

    /home/ubuntu/miniforge3/bin/conda init
    exit
fi

if [ -f "Miniforge3-$(uname)-$(uname -m).sh" ]; then
    rm "Miniforge3-$(uname)-$(uname -m).sh"
fi

if ! conda env list | grep -q "base"; then
    conda init bash
    source ~/.bashrc
    conda activate base
fi

# conda update -n base -c conda-forge conda

if ! conda env list | grep -q "shopgpt"; then
    conda env create -f ./environment.yaml
else
    conda env update -f ./environment.yaml
fi

# conda activate shopgpt

#pip install flash-attn --no-build-isolation
#conda install xformers -c xformers
#pip install "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git"