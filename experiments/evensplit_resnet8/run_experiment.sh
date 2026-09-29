#!/bin/bash
# One-shot driver for the equal-split ResNet-8-scratch experiment series.
# Run from the repo root (or anywhere -- it locates the root itself):
#   bash experiments/evensplit_resnet8/run_experiment.sh <n_generalists> <purity> [seed]
# e.g. bash experiments/evensplit_resnet8/run_experiment.sh 10 0.5 90
#
# Does: local training (equal generalist split, ResNet-8 scratch) -> logit generation ->
# full metrics (Table A/B) -> where-the-pipeline-loses breakdown -> other-classes trade-off check.
# All exist already as reusable scripts/analyses at the repo root; this just chains them and
# files the output under experiments/evensplit_resnet8/runs/<name>/ instead of scattering it.
set -e
root_dir="$(cd "$(dirname "$0")/../.." && pwd)"
env_python=/root/miniconda3/envs/fi/bin/python

n_generalists=${1:?number of generalists, e.g. 10}
purity=${2:?specialist_purity, e.g. 0.5}
seed=${3:-90}

name="g${n_generalists}_purity${purity}_seed${seed}"
out_dir="$root_dir/experiments/evensplit_resnet8/runs/$name"
mkdir -p "$out_dir"
cd "$root_dir"

echo "==> [1/2] local training + logit generation: $name"
bash scripts_submission/run_evensplit_resnet8.sh "$n_generalists" "$purity" "$seed" > "$out_dir/train.log" 2>&1

tag="_evensplit_g${n_generalists}_purity${purity}"
if [ "$seed" != "90" ]; then tag="${tag}_seed${seed}"; fi
size=$((n_generalists + 3))
logits="$root_dir/FL/results/logits/CIFAR10_0.5_M${size}_${seed}_0.1${tag}"

echo "==> [2/2] analysis -> $out_dir"
"$env_python" analysis_full_metrics.py "$name=$logits" > "$out_dir/full_metrics.log" 2>&1
mv results/C_meta_pipeline/fingerprint_validity/full_metrics_tableA.csv "$out_dir/tableA.csv"
mv results/C_meta_pipeline/fingerprint_validity/full_metrics_tableB.csv "$out_dir/tableB.csv"
"$env_python" analysis_where_specialist_loses.py "$name=$logits" > "$out_dir/where_specialist_loses.log" 2>&1
"$env_python" analysis_other_classes_check.py "$name=$logits" > "$out_dir/other_classes_check.log" 2>&1

echo "==> Done. Results in $out_dir"
