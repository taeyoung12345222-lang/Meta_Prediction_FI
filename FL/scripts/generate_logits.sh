#!/bin/bash

root_dir="/mnt/c/Users/황태영/OneDrive/Desktop/federated inference/robust-federated-inference-main/FL"
save_dir=logits

env_python=/root/miniconda3/envs/fi/bin/python

dataset=CIFAR10
datapath=/root/data
seeds=(90)
proxy_ratio=0.1
alphas=(0.5)
size=17

partition_method=specialist_mix
n_specialists=3
specialist_classes="3 4 9" # cat, deer, truck
specialist_purity=0.85
generalist_access_ratio=0.15

for alpha in ${alphas[@]}; do
    
    for seed in ${seeds[@]}; do

        modelpath="${root_dir}/results/local_training/${dataset}_localSGD_${alpha}_M${size}_${seed}_${proxy_ratio}"

        name=${dataset}_${alpha}_M${size}_${seed}_${proxy_ratio}
        log_dir="${root_dir}/results/${save_dir}/${name}"
        mkdir -p "$log_dir"

        # count time for experiment in hh mm ss
        start=`date +%s`

        "$env_python" "$root_dir/generate_logits_v2.py" \
            --model ViT_B32 \
            --dataset $dataset \
            --datapath "$datapath" \
            --modelpath "$modelpath" \
            --partition_method $partition_method \
            --n_specialists $n_specialists --specialist_classes $specialist_classes \
            --specialist_purity $specialist_purity --generalist_access_ratio $generalist_access_ratio \
            --save_dir "$log_dir" \
            --alpha $alpha \
            --seed $seed \
            --proxy_ratio $proxy_ratio \
            --batch_size 64 \
            --gpu

        end=`date +%s`
        runtime=$((end-start))
        # Print time in hh:mm:ss
        echo "==> Time taken: $(($runtime / 3600 )):$((($runtime / 60) % 60)):$(( $runtime % 60 ))"
    
    done

done
