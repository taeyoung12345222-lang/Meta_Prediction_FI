#!/bin/bash
# Experiment A: RFI original aggregator pipeline (no q_i method) on the specialist_mix logits.
# Follows the released test_Deepset_wb.sh: train on DeepSet_M (mean) with PGD adversarial training
# (cw loss, in-place adversaries), test later with DeepSet_TM2 (trimmed mean, trim 0.25).
#
# Usage (from the repo root, under WSL):
#   bash scripts_submission/train_A_rfi_original.sh                      # full run: n_subsets=300, 10 epochs
#   NSUBSETS=20 EPOCHS=1 TAG=_probe bash scripts_submission/train_A_rfi_original.sh   # timing probe

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
save_dir=A_rfi_original/agg_training/cifar10

env_python=/root/miniconda3/envs/fi/bin/python
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

dataset=CIFAR10
seed=90 # must match the seed used for local_training.sh / generate_logits.sh
proxy_ratio=0.1
alpha=0.5
size=17
model=DeepSet_M
trim_ratio=0.25
black_box=false
collude=true
attack_type=pgd
n_subsets=${NSUBSETS:-300}
epochs=${EPOCHS:-10}
tag=${TAG:-}

maxss=$size
minss=$((size-4)) # min set size = size - f (max number of attackers)

name=${dataset}_${alpha}_M${size}_${seed}_${proxy_ratio}_${model}_maxss${maxss}_minss${minss}_trim${trim_ratio}_attack${attack_type}_bb${black_box}_collude${collude}_nsubsets${n_subsets}_cw${tag}
log_dir="${root_dir}/results/${save_dir}/${name}"
mkdir -p "$log_dir"

datapath="${root_dir}/FL/results/logits/${dataset}_${alpha}_M${size}_${seed}_${proxy_ratio}"

start=`date +%s`

cd "$root_dir"
"$env_python" aggregator_training_fl.py \
    --size $size \
    --model $model \
    --dataset $dataset \
    --datapath "$datapath" \
    --partition_method dirichlet \
    --save_dir "$log_dir" \
    --alpha $alpha \
    --seed $seed \
    --proxy_ratio $proxy_ratio \
    --batch_size 256 \
    --gpu \
    --dim_hidden 224 \
    --debug \
    --normalize \
    --lr 5e-5 \
    --loss_fn cw \
    --epochs $epochs \
    --optimizer Adam \
    --add_subsets \
    --n_subsets $n_subsets \
    --max_set_size $maxss \
    --min_set_size $minss \
    --trim_ratio $trim_ratio \
    --adversarial \
    --attack_type $attack_type --collude
status=$?

end=`date +%s`
runtime=$((end-start))
echo "==> exit status: $status"
echo "==> Time taken: $(($runtime / 3600 )):$((($runtime / 60) % 60)):$(( $runtime % 60 ))"
echo "==> log_dir: $log_dir"
exit $status
