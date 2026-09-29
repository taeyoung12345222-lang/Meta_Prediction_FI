#!/bin/bash
# Experiment A evaluation: model trained by train_A_rfi_original.sh (DeepSet_M + PGD adv. training),
# tested as DeepSet_TM2 (trimmed mean, trim 0.25) as in the released test_Deepset_wb.sh.
#
# Usage (from the repo root, under WSL):
#   bash scripts_submission/test_A_rfi_original.sh                       # full model
#   NSUBSETS=20 TAG=_probe bash scripts_submission/test_A_rfi_original.sh   # validate on the probe model

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
env_python=/root/miniconda3/envs/fi/bin/python
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

dataset=CIFAR10
seed=90
proxy_ratio=0.1
alpha=0.5
size=17
train_model=DeepSet_M
test_model=DeepSet_TM2
trim_ratio=0.25
n_adv=4
n_generalists=14
specialist_classes="3 4 9"
n_subsets=${NSUBSETS:-300}
tag=${TAG:-}
attacks=(${ATTACKS:-sia lma pgd}) # cpa needs a similarity matrix (--S_path), bb attacks (dfl/sia-bb) are run separately

name2=${dataset}_${alpha}_M${size}_${seed}_${proxy_ratio}
datapath="${root_dir}/FL/results/logits/${name2}"
modelpath="${root_dir}/results/A_rfi_original/agg_training/cifar10/${name2}_${train_model}_maxss${size}_minss$((size-4))_trim${trim_ratio}_attackpgd_bbfalse_colludetrue_nsubsets${n_subsets}_cw${tag}"
out_root="${root_dir}/results/A_rfi_original"

cd "$root_dir"

echo "==> [1/2] Acc_all / Acc_rescue (clean) for static aggregators + trained DeepSet as ${test_model}"
"$env_python" evaluate_rescue.py \
    --datapath "$datapath" \
    --dataset $dataset \
    --models F_Avg F_Median2 F_Geo_Median F_TM2 $test_model \
    --modelpaths "$modelpath" \
    --n_generalists $n_generalists \
    --specialist_classes $specialist_classes \
    --trim_ratio $trim_ratio \
    --dim_hidden 224 \
    --normalize --normalization_type simplex \
    --gpu \
    --save_csv "${out_root}/A_rescue${tag}.csv" || exit 1

echo "==> [2/2] White-box attacks with n_adv=${n_adv} (in-place, colluding)"
for attack_type in "${attacks[@]}"; do
    log_dir="${out_root}/agg_testing/${name2}_${test_model}_trim${trim_ratio}_Nadv${n_adv}_TEST_attack${attack_type}_bbfalse_colludetrue${tag}"
    mkdir -p "$log_dir"
    echo "----- attack=${attack_type} -----"
    "$env_python" aggregator_testing_fl.py \
        --size $size \
        --model $test_model \
        --dataset $dataset \
        --datapath "$datapath" \
        --modelpath "$modelpath" \
        --partition_method dirichlet \
        --save_dir "$log_dir" \
        --alpha $alpha \
        --seed $seed \
        --proxy_ratio $proxy_ratio \
        --batch_size 64 \
        --dim_hidden 224 \
        --adversarial \
        --loss_fn cw \
        --eval_one_adv \
        --n_adv $n_adv \
        --trim_ratio $trim_ratio \
        --gpu \
        --normalize \
        --attack_type $attack_type --collude || exit 1
done

echo "==> Done. Results under ${out_root}"
