#!/bin/bash
configs=(
"./configs/models/unet_3.yaml"
"configs/models/adapt_unet.yaml"
"./configs/models/adapt_uctransnet.yaml"
"configs/models/uctransnet_3.yaml"
"./configs/models/attunet.yaml"
"./configs/models/resunet.yaml"
"configs/models/missformer.yaml"
"configs/models/transunet.yaml"
"configs/models/unetpp.yaml"
"configs/models/multiresunet.yaml"
"./configs/models/acda_uctransnet.yaml"
)
mkdir -p logs


for conf in "${configs[@]}"; do
    while : ; do
        srun -N 1-1 --gpus=1 nvidia-smi
        if [ $? -eq 0 ]; then
            name=$(basename "$conf" .yaml)
            current_date=$(date +"%Y%m%d-%H%M%S")
            log="logs/%j_${name}_${current_date}.log"
            sbatch -N 1-1 -J "$name" --gpus=1  --output="$log" --error="$log" adaptive_mis --dataset .\configs\datasets\isic.yaml  --model $conf
            break
        fi
        echo "no free gpu"
        sleep 10
        
        
    done
done