# Robust Federated Inference

![illustration](rens.drawio.svg)

This repository includes the code for our paper titled **"Robust Federated Inference"**, accepted at **ICLR'26**.
The goal of this project is to aggregate predictions from an ensemble of clients in a robust manner.

## Installation

The code is tested on Python 3.8. The dependencies can be installed using the following command. We use the name `fi` for federated inference.

```bash
conda create -n fi python=3.8
conda activate fi
pip install -r requirements.txt
```

## Setup for all experiments

The scripts directory includes sample run scripts. We provide an example below. The first step is to execute client local training. This is followed by logit generation which generates the data for the aggregator training task. 

```bash
# EXECUTE LOCAL TRAINING

# Update root_dir to point to the root directory of this repository on your machine
# Update datapath to point to the directory where the [CIFAR-10, CIFAR-100, AGNews] dataset is stored on your machine
# Update env_python to point to the python created above
# This could be something like /home/user/.conda/envs/fi/bin/python

# Please review FL/args.py for the list of possible args and then run
FL/scripts/local_training.sh
```

The local training runs parallely for all clients using pyTorch Distributed. Hence, the number of clients should be less than the total available CPU cores on the machine. The GPUs are equally split among the clients since each client does not use the entire GPU. The local training generates .pth model files, log files and train and test csvs per client in `results/local_training`. The models files will be used in the next step to generate logits.

```bash
# GENERATE LOGITS

# Update root_dir, env_python and datapath as above

# Please review FL/args.py for the list of possible args and then run
FL/scripts/generate_logit.sh
```
The logits are then generated for all clients and stored in the `results/logits` directory. These logits will be used in the next step to train and test the aggregation methods.

## Running DeepSet aggregator

Please ensure that the logits are generated as per the above step before running the DeepSet aggregator training and testing.

```bash
# TRAIN DeepSet AGGREGATOR

# Update root_dir, env_python and datapath as above

# Please review Utils/fl_args.py for the list of possible args and then run
/scripts/train_DeepSet.sh
```

The DeepSet aggregator is trained and the `.pth` model file with training logs are stored in `results/agg_training_adv`. 

```bash
# TEST DeepSet AGGREGATOR

# Update root_dir, env_python and datapath as above
# Please review Utils/fl_args.py for the list of possible args and then run

# For testing white-box attacks: lma, sia, cpa, pgd
/scripts/test_DeepSet_wb.sh

# For testing black-box attacks: dfl, sia
/scripts/test_DeepSet_bb.sh
```

The test results are stored in `results/agg_testing_adv`. The test log files include the accuracy for `f=0` and `f=n_adv` as specified in the script.

## Running Baselines

Please ensure that the logits are generated as per the above step before running the baselines training and testing.

```bash
# TEST Robust AGGREGATORS: Trimmed Mean, Median, Geometric Median

# Update root_dir, env_python and datapath as above
# Please review Utils/fl_args.py for the list of possible args and then run

# For testing white-box attacks: lma, sia, cpa, pgd
/scripts/test_F_TM_wb.sh

# For testing black-box attacks: dfl, sia
/scripts/test_F_TM_bb.sh
```

The test results are stored in `results/agg_testing`. The test log files include the accuracy for `f=0` and `f=n_adv` as specified in the script.

### CoPur training

We implement CoPur based on the code provided by the authors [here](https://github.com/AI-secure/CoPur/tree/main).

The first step is to the train the AutoEncoder model as well as the Server model in CoPur. 

```bash
# TRAIN CoPur AutoEncoder and Server model

# Update root_dir, env_python and datapath as above
# Please review Baselines/CoPur/args.py for the list of possible args

# To train the AutoEncoder
/scripts/train_ae.sh

# To train the Server model
/scripts/train_server_model.sh
```

The `.pth` files of the trained models are stored in `results/baselines/copur`.

### CoPur testing

```bash
# TEST CoPur

# Update root_dir, env_python and datapath as above
# Please review Baselines/CoPur/args.py for the list of possible args
/scripts/test_copur.sh
```
The test results are stored in `results/baselines/copur`. The test log files include the accuracy for `f=0` and `f=n_adv` (`n_adv` as specified in the script) for both CoPur as well as Manifold Projection.

## Citation

If you are using the code from this repository in your research, please consider citing our paper:

```
@inproceedings{
dhasade2026robust,
title={Robust Federated Inference},
author={Akash Dhasade and Sadegh Farhadkhani and Rachid Guerraoui and Nirupam Gupta and Maxime Jacovella and Anne-Marie Kermarrec and Rafael Pinot},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=47eKYCaBIV}
}
```
