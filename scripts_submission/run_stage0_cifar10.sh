#!/bin/bash
# Stage 0 end-to-end: specialist/generalist local training -> logit generation ->
# DeepSet variants training -> Acc_all / Acc_rescue table for every aggregator (statistical + DeepSet).
set -e
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH # cuDNN's dlopen("libcuda.so") can't find it otherwise

root_dir="/mnt/c/Users/황태영/OneDrive/Desktop/federated inference/robust-federated-inference-main"
env_python=/root/miniconda3/envs/fi/bin/python

dataset=CIFAR10
size=17
alpha=0.5
seed=90
proxy_ratio=0.1
n_generalists=14
specialist_classes="3 4 9" # cat, deer, truck
trim_ratio=0.25

echo "==> [1/5] Local training (17 clients: 14 generalist + 3 specialist)"
bash "$root_dir/FL/scripts/local_training.sh"

echo "==> [2/5] Generating logits"
bash "$root_dir/FL/scripts/generate_logits.sh"

echo "==> [3/5] Training DeepSet (plain, RFI original impl)"
bash "$root_dir/scripts_submission/train_DeepSet_plain.sh"

echo "==> [4/5] Training DeepSet_TM + adversarial training (RFI Table 1 final model)"
bash "$root_dir/scripts_submission/train_DeepSet.sh"

echo "==> [5/5] Evaluating Acc_all / Acc_rescue for all aggregators"

logits_dir=${root_dir}/FL/results/logits/${dataset}_${alpha}_M${size}_${seed}_${proxy_ratio}
maxss=$size
minss=$((size-4))

deepset_plain_dir=${root_dir}/results/stage0/agg_training/cifar10/${dataset}_${alpha}_M${size}_${seed}_${proxy_ratio}_DeepSet_maxss${maxss}_minss${minss}_nsubsets300_clean
deepset_tm_dir=${root_dir}/results/stage0/agg_training/cifar10/${dataset}_${alpha}_M${size}_${seed}_${proxy_ratio}_DeepSet_TM_maxss${maxss}_minss${minss}_trim${trim_ratio}_attackpgd_bbfalse_colludetrue_nsubsets300_cw

"$env_python" "$root_dir/evaluate_rescue.py" \
    --datapath "$logits_dir" \
    --dataset $dataset \
    --models F_Avg F_Median2 F_Geo_Median F_TM2 DeepSet DeepSet_TM \
    --modelpaths "$deepset_plain_dir" "$deepset_tm_dir" \
    --n_generalists $n_generalists \
    --specialist_classes $specialist_classes \
    --trim_ratio $trim_ratio \
    --dim_hidden 224 \
    --normalize --normalization_type simplex \
    --gpu \
    --save_csv "$root_dir/results/stage0/stage0_results.csv"

echo "==> Done. Results saved to $root_dir/results/stage0/stage0_results.csv"
