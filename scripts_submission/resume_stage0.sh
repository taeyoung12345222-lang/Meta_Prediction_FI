#!/bin/bash
set -e
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

ROOT="/mnt/c/Users/황태영/OneDrive/Desktop/federated inference/robust-federated-inference-main"

echo "==> [3/5] DeepSet plain already trained, skipping"

echo "==> [4/5] Training DeepSet_TM plus adversarial (n_subsets=20)"
bash "$ROOT/scripts_submission/train_DeepSet.sh"

echo "==> [5/5] Evaluating"
/root/miniconda3/envs/fi/bin/python "$ROOT/evaluate_rescue.py" \
    --datapath "$ROOT/FL/results/logits/CIFAR10_0.5_M17_90_0.1" \
    --dataset CIFAR10 \
    --models F_Avg F_Median2 F_Geo_Median F_TM2 DeepSet DeepSet_TM \
    --modelpaths "$ROOT/results/stage0/agg_training/cifar10/CIFAR10_0.5_M17_90_0.1_DeepSet_maxss17_minss13_nsubsets300_clean" "$ROOT/results/stage0/agg_training/cifar10/CIFAR10_0.5_M17_90_0.1_DeepSet_TM_maxss17_minss13_trim0.25_attackpgd_bbfalse_colludetrue_nsubsets20_cw" \
    --n_generalists 14 \
    --specialist_classes 3 4 9 \
    --trim_ratio 0.25 \
    --dim_hidden 224 \
    --normalize --normalization_type simplex \
    --gpu \
    --save_csv "$ROOT/results/stage0/stage0_results.csv"

echo "==> Done."
