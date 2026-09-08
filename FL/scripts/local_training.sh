#!/bin/bash

size=17 #generalist + 3 specialists
clr=0.01
slr=1.0
mu=0.0
momentum=0.0
gmf=0.0
proxy_ratio=0.1 # keep 10% data as for aggregator training
optimizer=localSGD
dataset=CIFAR10
datapath=/root/data

root_dir="/mnt/c/Users/황태영/OneDrive/Desktop/federated inference/robust-federated-inference-main/FL"
save_dir=local_training
log_dir="$root_dir/results/$save_dir"
mkdir -p "$log_dir/$name"

env_python=/root/miniconda3/envs/fi/bin/python

alphas=(0.5)
seeds=(90)
le=20
global_rounds=1

# specialist/generalist mixed partition (Stage 0: cat/deer/truck specialists vs the rest generalists)
partition_method=specialist_mix
n_specialists=3
specialist_classes="3 4 9" # cat, deer, truck -- each confusable with a class generalists see plenty of (dog, horse, automobile)
specialist_purity=0.85
generalist_access_ratio=0.15

for alpha in ${alphas[@]}; do
    
    for seed in ${seeds[@]}; do

        name=${dataset}_${optimizer}_${alpha}_M${size}_${seed}_${proxy_ratio}

        mkdir -p "$log_dir/$name"
        echo "==> Running $name"

        # count time for experiment in hh mm ss
        start=`date +%s`

        # optimizer=localSGD does no cross-client communication (see train_LocalSGD.py's
        # maybe_barrier()/init_processes), so clients don't need to be simultaneously resident
        # in memory -- run them in small batches to keep peak RAM manageable.
        batch_size=4
        fail=0
        for batch_start in $(seq 0 $batch_size $((size-1))); do

            batch_end=$((batch_start + batch_size - 1))
            if [ $batch_end -gt $((size-1)) ]; then batch_end=$((size-1)); fi
            echo "==> Batch: ranks $batch_start-$batch_end"

            pids=()
            for rank in $(seq $batch_start $batch_end); do

                OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LD_LIBRARY_PATH=/usr/lib/wsl/lib "$env_python" "$root_dir/train_LocalSGD.py" --lr $clr --slr $slr --bs 16 --mu $mu \
                                    --lowE $le --highE $le -iE --use_scheduler \
                                    --momentum $momentum --gmf $gmf \
                                    --numclients $size --rank $rank --size $size --totalclients $size \
                                    --backend gloo --initmethod tcp://localhost:23000 \
                                    --weights data_based --diff_init \
                                    --rounds $global_rounds --seed $seed --NIID --print_freq 100 \
                                    --save --name $name --disable_wandb \
                                    --partition_method $partition_method --alpha $alpha \
                                    --n_specialists $n_specialists --specialist_classes $specialist_classes \
                                    --specialist_purity $specialist_purity --generalist_access_ratio $generalist_access_ratio \
                                    --dataset $dataset --optimizer $optimizer --model ViT_B32 \
                                    --savepath "$log_dir" --evalafter 5 --test_bs 16 \
                                    --datapath "$datapath" \
                                    --proxy_set --proxy_ratio $proxy_ratio \
                                    -sm -g --procs_per_machine -1 &
                pids+=($!)
                sleep 3 # avoid simultaneous CIFAR10/model-checkpoint load within a batch
            done

            for pid in "${pids[@]}"; do
                wait "$pid" || fail=1
            done
        done
        if [ "$fail" -ne 0 ]; then
            echo "==> ERROR: at least one client training process failed, aborting"
            exit 1
        fi

        end=`date +%s`
        runtime=$((end-start))
        # Print time in hh:mm:ss
        echo "==> Time taken: $(($runtime / 3600 )):$((($runtime / 60) % 60)):$(( $runtime % 60 ))"

    done
done