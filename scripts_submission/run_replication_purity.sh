#!/bin/bash
# ResNet-8 (scratch) clients on the seed-90 partition with a different specialist purity.
#   bash scripts_submission/run_replication_purity.sh 0.45
# Same recipe as run_replication.sh resnet8 (SGD lr 0.0025, momentum 0.9, bs 16, 100 epochs, no scheduler, 32x32 inputs).
set -u
root_dir="$(cd "$(dirname "$0")/.." && pwd)"
env_python=/root/miniconda3/envs/fi/bin/python
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

dataset=CIFAR10; datapath=/root/data; size=17; alpha=0.5; proxy_ratio=0.1; seed=90
n_specialists=3; specialist_classes="3 4 9"; generalist_access_ratio=0.15
specialist_purity=${1:?purity, e.g. 0.45}
model=ResNet8; tag="_resnet8_purity${specialist_purity}"; lr=0.0025; mom=0.9; bs=16; epochs=100; img=32; test_bs=64; logit_bs=256; parallel=6

name=${dataset}_localSGD_${alpha}_M${size}_${seed}_${proxy_ratio}${tag}
train_root="$root_dir/FL/results/local_training"
mkdir -p "$train_root/$name"
echo "==> [resnet8 purity $specialist_purity] local training: $name"
start=$(date +%s)

all_ranks=($(seq 0 $((size-1))))
fail=0
for ((i=0; i<${#all_ranks[@]}; i+=parallel)); do
    pids=()
    for rank in "${all_ranks[@]:i:parallel}"; do
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$env_python" "$root_dir/FL/train_LocalSGD.py" \
            --lr $lr --slr 1.0 --bs $bs --mu 0.0 --lowE $epochs --highE $epochs -iE \
            --momentum $mom --gmf 0.0 \
            --numclients $size --rank $rank --size $size --totalclients $size \
            --backend gloo --initmethod tcp://localhost:23000 \
            --weights data_based --diff_init --rounds 1 --seed $seed --NIID --print_freq 100 \
            --save --name "$name" --disable_wandb \
            --partition_method specialist_mix --alpha $alpha \
            --n_specialists $n_specialists --specialist_classes $specialist_classes \
            --specialist_purity $specialist_purity --generalist_access_ratio $generalist_access_ratio \
            --dataset $dataset --optimizer localSGD --model $model \
            --savepath "$train_root" --evalafter 5 --test_bs $test_bs \
            --datapath "$datapath" --img_size $img \
            --proxy_set --proxy_ratio $proxy_ratio \
            -sm -g --procs_per_machine -1 &
        pids+=($!)
        sleep 3
    done
    for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
done
if [ "$fail" -ne 0 ]; then echo "==> ERROR: a client training process failed"; exit 1; fi
echo "==> local training done in $(( $(date +%s) - start )) s"

lname=${dataset}_${alpha}_M${size}_${seed}_${proxy_ratio}${tag}
log_dir="$root_dir/FL/results/logits/$lname"
mkdir -p "$log_dir"
echo "==> generating logits: $lname"
"$env_python" "$root_dir/FL/generate_logits_v2.py" \
    --model $model --dataset $dataset --datapath "$datapath" --modelpath "$train_root/$name" \
    --partition_method specialist_mix --n_specialists $n_specialists --specialist_classes $specialist_classes \
    --specialist_purity $specialist_purity --generalist_access_ratio $generalist_access_ratio \
    --save_dir "$log_dir" --alpha $alpha --seed $seed --proxy_ratio $proxy_ratio \
    --batch_size $logit_bs --img_size $img --gpu || { echo "==> ERROR: logit generation failed"; exit 1; }
echo "==> [resnet8 purity $specialist_purity] finished in $(( $(date +%s) - start )) s"
