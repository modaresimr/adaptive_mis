name=$(basename "$conf" .yaml)
current_date=$(date +"%Y%m%d-%H%M%S")
save_dir="results/$(basename $dataset .yaml)/${name}_${current_date}"
mkdir -p $save_dir
log="$save_dir/out_%j.log"
echo srun -N 1-1 -J "$name" --gpus=1  --output="$log" --error="$log" adaptive_mis --dataset $dataset --save_dir $save_dir --model $conf
sbatch -N 1-1 -J "$name" --gpus=1  --output="$log" --error="$log"  ./run.sh --dataset $dataset --save_dir $save_dir --model $conf
