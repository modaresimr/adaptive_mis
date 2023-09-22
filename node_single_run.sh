#!/bin/bash
dataset=$1
conf=$2
name=$(basename "$conf" .yaml)
current_date=$(date +"%Y%m%d-%H%M%S")
save_dir="results/$(basename $dataset .yaml)/${name}_${current_date}"
mkdir -p $save_dir
log="$save_dir/out_%j.log"

options="-N 1-1 -J $name --gpus=1"

runcmd="./run.sh --dataset $dataset --save_dir $save_dir --model $conf --eval configs/evaluation/split.yaml"
echo srun $options $runcmd
options="$options --output=$log --error=$log "
sbatch $options $runcmd
