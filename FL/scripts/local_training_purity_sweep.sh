#!/bin/bash
# Purity sweep: same pipeline as local_training.sh (Stage 0's specialist_mix setup),
# but varies specialist_purity while holding everything else (seed, alpha, specialist_classes,
# generalist_access_ratio) fixed, to check whether the q_i / Acc_rescue findings scale
# continuously with specialization strength rather than being an artifact of purity=0.85 alone.
# purity=0.85 is NOT re-run here -- it's already trained under FL/results/local_training/CIFAR10_localSGD_0.5_M17_90_0.1.

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

env_python=/root/miniconda3/envs/fi/bin/python

alphas=(0.5)
seeds=(90)
le=20
global_rounds=1

# specialist/generalist mixed partition (Stage 0: cat/deer/truck specialists vs the rest generalists)
partition_method=specialist_mix
n_specialists=3
specialist_classes="3 4 9" # cat, deer, truck
generalist_access_ratio=0.15
purities=(0.65 0.45 0.25) # 0.85 already trained separately

for purity in ${purities[@]}; do

    specialist_purity=$purity

    for alpha in ${alphas[@]}; do

        for seed in ${seeds[@]}; do

            name=${dataset}_${optimizer}_${alpha}_M${size}_${seed}_${proxy_ratio}_purity${specialist_purity}

            mkdir -p "$log_dir/$name"
            echo "==> Running $name"

            start=`date +%s`

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
                    sleep 3
                done

                for pid in "${pids[@]}"; do
                    wait "$pid" || fail=1
                done
            done
            if [ "$fail" -ne 0 ]; then
                echo "==> ERROR: at least one client training process failed for purity=$specialist_purity, aborting"
                exit 1
            fi

            end=`date +%s`
            runtime=$((end-start))
            echo "==> [purity=$specialist_purity] Time taken: $(($runtime / 3600 )):$((($runtime / 60) % 60)):$(( $runtime % 60 ))"

        done
    done
done
