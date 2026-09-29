#!/bin/bash
# Purity sweep end-to-end: local training -> logit generation -> Stage 1 (q_i validity)
# and Stage 0-lite (static aggregators only, no DeepSet retraining) evaluation, for
# specialist_purity in {0.65, 0.45, 0.25}. purity=0.85 already exists from the original
# Stage 0/1 run and is included in the summary via --extra_purity085.
set -e
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

root_dir="/mnt/c/Users/황태영/OneDrive/Desktop/federated inference/robust-federated-inference-main"
env_python=/root/miniconda3/envs/fi/bin/python

dataset=CIFAR10
size=17
alpha=0.5
seed=90
proxy_ratio=0.1
n_generalists=14
specialist_classes="3 4 9"
trim_ratio=0.25

purities=(0.65 0.45 0.25)

echo "==> [1/3] Local training for purity in {${purities[@]}} (0.85 already done)"
bash "$root_dir/FL/scripts/local_training_purity_sweep.sh"

echo "==> [2/3] Generating logits for purity in {${purities[@]}}"
bash "$root_dir/FL/scripts/generate_logits_purity_sweep.sh"

echo "==> [3/3] Stage 1 (q_i validity) + Stage 0-lite (static aggregators) evaluation per purity"
mkdir -p "$root_dir/results/purity_sweep"

for purity in "${purities[@]}"; do
    logits_dir="${root_dir}/FL/results/logits/${dataset}_${alpha}_M${size}_${seed}_${proxy_ratio}_purity${purity}"

    echo "----- purity=$purity -----"
    "$env_python" "$root_dir/evaluate_qi.py" \
        --datapath "$logits_dir" \
        --dataset $dataset \
        --n_generalists $n_generalists \
        --specialist_classes $specialist_classes \
        --save_csv "$root_dir/results/purity_sweep/stage1_purity${purity}.csv"

    "$env_python" "$root_dir/evaluate_rescue.py" \
        --datapath "$logits_dir" \
        --dataset $dataset \
        --models F_Avg F_Median2 F_Geo_Median F_TM2 \
        --n_generalists $n_generalists \
        --specialist_classes $specialist_classes \
        --trim_ratio $trim_ratio \
        --normalize --normalization_type simplex \
        --gpu \
        --save_csv "$root_dir/results/purity_sweep/stage0lite_purity${purity}.csv"
done

echo "==> Done. Per-purity results in $root_dir/results/purity_sweep/"
echo "==> Reference (purity=0.85, already trained): results/stage1/*.csv and results/stage0/stage0_results.csv"
