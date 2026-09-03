#!/bin/bash

root_dir=<absolute_path_to_the_root_of_this_repository>
save_dir=agg_testing/cifar100

env_python=<point_to_your_conda_python_interpreter>

dataset=CIFAR100
datapath=<point_to_your_dataset_path>
seeds=(90)
proxy_ratio=0.1
alphas=(0.5)
model=F_TM2 # F_Avg, F_Median2, F_Geo_Median
train_model=F_TM2
trim_ratio=0.2
sizes=(17)
n_advs=(4)
black_box=true
collude=false
maxss=$((sizes[0]))
minss=$((sizes[0]-4)) # set min set size to size - f (i.e. max number of attackers)
attacks=(lma)

if [ "$black_box" = true ]; then
    black_box_str="--black_box"
else
    black_box_str=""
fi

if [ "$collude" = true ]; then
    collude_str="--collude"
else
    collude_str=""
fi

for size in "${sizes[@]}"; do
    
    for alpha in "${alphas[@]}"; do

        for seed in "${seeds[@]}"; do

            for attack_type in "${attacks[@]}"; do

                for n_adv in "${n_advs[@]}"; do

                    echo "==> Running test for n_adv: $n_adv, alpha: $alpha, seed: $seed"

                    name2=${dataset}_${alpha}_M${size}_${seed}_${proxy_ratio}
                    datapath=${root_dir}/results/logits/${name2}

                    name3=${dataset}_${alpha}_M${size}_${seed}_${proxy_ratio}_${train_model}_maxss${maxss}_minss${minss}_trim${trim_ratio}
                    modelpath=${root_dir}/results/agg_training/cifar100/${name3}

                    name=${dataset}_${alpha}_M${size}_${seed}_${proxy_ratio}_${model}_adv_trim${trim_ratio}_Nadv${n_adv}_TEST_attack${attack_type}_bb${black_box}_collude${collude}_TRAIN_attackpgd_bbfalse_colludetrue
                    log_dir=${root_dir}/results/${save_dir}/${name}
                    
                    mkdir -p $log_dir

                    # count time for experiment in hh mm ss
                    start=`date +%s`

                    $env_python $root_dir/aggregator_testing_fl.py \
                        --size $size \
                        --model $model \
                        --dataset $dataset \
                        --datapath $datapath \
                        --modelpath $modelpath \
                        --partition_method dirichlet \
                        --save_dir $log_dir \
                        --alpha $alpha \
                        --seed $seed \
                        --proxy_ratio $proxy_ratio \
                        --batch_size 64 \
                        --adversarial \
                        --loss_fn cw \
                        --eval_one_adv \
                        --n_adv $n_adv \
                        --trim_ratio $trim_ratio \
                        --gpu \
                        --normalize \
                        --attack_type $attack_type $black_box_str $collude_str

                    end=`date +%s`
                    runtime=$((end-start))
                    # Print time in hh:mm:ss
                    echo "==> Time taken: $(($runtime / 3600 )):$((($runtime / 60) % 60)):$(( $runtime % 60 ))"

                done
            done
        done
    done
done
