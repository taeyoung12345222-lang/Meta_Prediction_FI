# Code for Fens

![fens2](https://github.com/user-attachments/assets/a7a1575a-38fa-4386-ba2c-028fbf075840)

This respository contains the code for our paper: **"Revisiting Ensembling in One-Shot Federated Learning", NeurIPS 2024**. This code was inspired from the [FedNova code repository](https://github.com/JYWa/FedNova).

## Installation

The code is tested on Python 3.9. The dependencies can be installed using the following command.

```bash
conda create -n fens python=3.9
conda activate fens
pip install -r requirements.txt
```
## Wandb

This code repository uses [wandb](https://wandb.ai/) for logging. Please create an account on wandb and login using the following command.

```bash
wandb login
```

Do not forget to update the `wandb` default configuration in the `args.py` and `flamby/args.py` file. Specifically, update the `wandb_project` and `wandb_entity` to your project and entity name respectively. They could also be passed as arguments to all scripts described later in the format shown below.

```bash
--wandb_project <name> --wandb_entity <name>
```

## Running Fens Experiments

The `scripts` directory includes sample run scripts. We provide an example below. In order to run Fens, we first execute local training. This is followed by logit generation which generates the data for the aggregator training task. Finally, we train the aggregator model using FL. These three scripts can be executed as follows:

```bash
# EXECUTE LOCAL TRAINING

# Update root_dir to point to the root directory of this repository on your machine
# Update datapath to point to the directory where the [CIFAR-10, CIFAR-100, SVHN] dataset is stored on your machine
# Update env_python to point to the python created above
# This could be something like /home/user/.conda/envs/fens/bin/python

# Please review args.py for the list of possible args and then run
fens/scripts/local_training.sh
```

The local training runs parallely for all clients using pyTorch Distributed. Hence, the number of clients should be less than the total available CPU cores on the machine. The GPUs are equally split among the clients since each client does not use the entire GPU. The local training generates `.pth` model files, log files and train and test csvs per client in `results/local_training`. The models files will be used in the next step to generate logits.

```bash
# GENERATE LOGITS

# Update root_dir to point to the root directory of this repository on your machine
# Update datapath to point to the directory where the [CIFAR-10, CIFAR-100, SVHN] dataset is stored on your machine
# Update env_python to point to the python created above

# Please review args.py for the list of possible args and then run
fens/scripts/generate_logit.sh
``` 
This script first reads the models and quantises them from FP32 to INT8. The logits are then generated for all clients and stored in the `results/logits` directory. These logits will be used in the next step to train the aggregator model.

```bash
# EXECUTE AGGREGATOR TRAINING

# Update root_dir to point to the root directory of this repository on your machine
# Update datapath to point to the directory where the [CIFAR-10, CIFAR-100, SVHN] dataset is stored on your machine
# Update env_python to point to the python created above

# Please review args.py for the list of possible args and then run
fens/scripts/train_aggregator.sh
```

The aggregator training runs parallely for all clients using pyTorch Distributed. This is very similar to the local training, except it uses a different `distoptim` optimizer. Checkout `fens/distoptim` for all available distributed optimisers which are FL algorithms. The execution produces the final global, log files and train and test csvs per client in `results/aggregator` directory.

## Fens FLamby experiments

The FLamby experiments in Fens are built on top of the benchmark codebase [Flamby](https://github.com/owkin/FLamby). Please follow the instructions in the Flamby repository to setup the python environment and the three datasets including `Fed-ISIC2019`, `Fed-Camelyon16` and `Fed-Heart-Disease`. After the setup and activation of flamby conda environment, the following line should execute without errors. 

```python
from flamby.datasets.fed_isic2019 import (
            BATCH_SIZE,
            LR,
            NUM_EPOCHS_POOLED,
            NUM_CLIENTS,
            Optimizer,
            Baseline,
            BaselineLoss,
            FedIsic2019 as FedDataset
        )
``` 

We follow the same procedure as above to run the Fens experiments. The only difference is that the local training script also generates logits for the aggregator training. 

```bash
# EXECUTE LOCAL TRAINING AND GENERATE LOGITS

# Update root_dir to point to the root directory of this repository on your machine
# There is no datapath here as it will set during the flamby setup
# Update env_python to point to the conda flamby environment created in the flamby setup

# Please review flamby/args.py for the list of possible args and then run
fens/scripts/flamby/fedisic_local_training.sh
```

The local training here is sequential for each client. The logits are generated for all clients and stored in the `results/flamby/local_training` directory along with the log files. These logits will be used in the next step to train the aggregator model.

```bash
# EXECUTE AGGREGATOR TRAINING

# Update root_dir to point to the root directory of this repository on your machine
# There is no datapath here as the logitpath will serve as the datapath
# Update env_python to point to the conda fens environment created above
# Note that the aggregator training should be run in conda fens environment

# Please review args.py for the list of possible args and then run
fens/scripts/flamby/fedisic_agg_training.sh
```

This aggregator training is similar the aggregator training for non flamby datasets. The execution produces the final global, log files and train and test csvs per client in `results/flamby/aggregator` directory.

## Iterative FL baselines

```bash
# EXECUTE FedAdam BASELINE

# Update root_dir to point to the root directory of this repository on your machine
# Update datapath to point to the directory where the [CIFAR-10, CIFAR-100, SVHN] dataset is stored on your machine
# Update env_python to point to the python created above

# Please review args.py for the list of possible args and then run
fens/scripts/fl_train.sh
```

The `fl_train.sh` script provide an example run of the `FedAdam` baseline for the CIFAR-10 dataset. The script generates the final global, log files and train and test csvs per client in `results/fl_training` directory. This script can be modified to run the other iterative FL baselines, present in `distoptim` directory.

## One-shot FL baselines

The one-shot FL baselines were run using the original code repositories of the respective papers. The code repositories are referenced below for ease of access. 

1. [Co-Boosting](https://github.com/rong-dai/Co-Boosting) (ICLR' 24)
2. [FedCVAE-Ens](https://github.com/ceh-2000/fed_cvae) (ICLR' 23)
3. [FedKD](https://github.com/gong-xuan/FedKD) (AAAI' 22)
