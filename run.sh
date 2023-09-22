#!/bin/bash
#pip install -e .
if [[ ":$PATH:" != *":/users/modaresi/.local/bin:"* ]]; then
    export PATH="/users/modaresi/.local/bin:$PATH"
fi
stty cols 130
#stty size | awk '{print $2}'
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
adaptive_mis $@
