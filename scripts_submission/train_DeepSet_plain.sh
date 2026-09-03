#!/bin/bash
# "DeepSet Aggregator, RFI 원 구현" -- plain DeepSet, no adversarial training, no trimmed mean.
# Companion to train_DeepSet.sh (DeepSet_TM + adversarial training, the RFI Table 1 final model).

root_dir="/mnt/c/Users/황태영/OneDrive/Desktop/federated inference/robust-federated-inference-main"
save_dir=stage0/agg_training/cifar10

env_python=/root/miniconda3/envs/fi/bin/python

dataset=CIFAR10
seeds=(90) # must match the seed used for local_training.sh / generate_logits.sh
proxy_ratio=0.1
alphas=(0.5)
model=DeepSet
sizes=(17)
n_subsets=300

for size in "${sizes[@]}"; do

    for alpha in "${alphas[@]}"; do

        for seed in "${seeds[@]}"; do

            maxss=$((size))
            minss=$((size-4))

            name=${dataset}_${alpha}_M${size}_${seed}_${proxy_ratio}_${model}_maxss${maxss}_minss${minss}_nsubsets${n_subsets}_clean
            log_dir="${root_dir}/results/${save_dir}/${name}"
            mkdir -p "$log_dir"

            name2=${dataset}_${alpha}_M${size}_${seed}_${proxy_ratio}
            datapath="${root_dir}/FL/results/logits/${name2}"

            start=`date +%s`

            "$env_python" "$root_dir/aggregator_training_fl.py" \
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
                --loss_fn ce \
                --epochs 10 \
                --optimizer Adam \
                --add_subsets \
                --n_subsets $n_subsets \
                --max_set_size $maxss \
                --min_set_size $minss

            end=`date +%s`
            runtime=$((end-start))
            echo "==> Time taken: $(($runtime / 3600 )):$((($runtime / 60) % 60)):$(( $runtime % 60 ))"

        done
    done
done
