import argparse

def get_copur_args():
    """
    Get arguments for CoPur baseline, extending the base FL arguments
    with CoPur-specific parameters.
    """
    parser = argparse.ArgumentParser(description='CoPur Autoencoder Baseline')
    
    # Dataset arguments
    parser.add_argument('--size',
                        default=20,
                        type=int,
                        help='number of clients')
    parser.add_argument('--dataset',
                        default='CIFAR10',
                        type=str,
                        choices=['CIFAR10', 'AG_News', 'CIFAR100'],
                        help='which dataset to run on')
    parser.add_argument('--datapath',
                        required=True,
                        type=str,
                        help='directory where the generated logits are stored')
    parser.add_argument('--modelpath',
                        default=None,
                        type=str,
                        help='directory where the trained autoencoder is saved')
    parser.add_argument('--modelpath2',
                        default=None,
                        type=str,
                        help='directory where the trained server model is saved')
    parser.add_argument('--partition_method',
                        default='dirichlet',
                        choices=['dirichlet', 'sharding'],
                        type=str,
                        help='how to partition the dataset')
    parser.add_argument('--save_dir',
                        required=True,
                        type=str,
                        help='directory to save outputs')
    parser.add_argument('--alpha',
                        default=0.1,
                        type=float,
                        help='alpha for dirichlet distribution')
    parser.add_argument('--n_classes_per_client',
                        default=4,
                        type=int,
                        help='number of classes per client for sharding based heterogeneity')
    parser.add_argument('--seed',
                        default=90,
                        type=int,
                        help='seed for random number generators')
    parser.add_argument('--gpu',
                        action='store_true',
                        help='whether to use GPU')
    
    # Training parameters
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='learning rate')
    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        help='number of epochs')
    parser.add_argument('--optimizer',
                        default='Adam',
                        choices=['SGD', 'Adam'],
                        type=str,
                        help='optimizer')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='batch size for training')
    
    # Attack
    parser.add_argument('--attack_type',
                        choices=['dfl', 'sia', 'none'],
                        default='none',
                        help='which attack to run, dfl: distributed feature flipping')
    parser.add_argument('--black_box',
                        action='store_true',
                        help='if true, adversaries are black-box | if false, adversaries are white-box')
    parser.add_argument('--amplification',
                        type=float,
                        default=10,
                        help='amplification factor for DFL or SIA attack')
    parser.add_argument('--n_adv',
                        default=2,
                        type=int,
                        help='number of adversaries')
    parser.add_argument('--eval_one_adv',
                        action='store_true',
                        help='whether to evaluate only for n_adv adversaries and not all values from [1, n_adv]')
    
    # Logging
    parser.add_argument('--debug',
                        action='store_true',
                        help='whether to log debug messages')
    parser.add_argument('--wandb',
                        action='store_true',
                        help='Use wandb')
    parser.add_argument('--wandb_project',
                        type=str,
                        default='',
                        help='wandb project name')
    parser.add_argument('--wandb_entity',
                        type=str,
                        default='',
                        help='wandb entity name')
    
    # CoPur-specific arguments
    parser.add_argument('--hidden_dim',
                        default=200,
                        type=int,
                        help='hidden dimension for autoencoder')
    parser.add_argument('--encode_dim',
                        default=120,
                        type=int,
                        help='encoding dimension for autoencoder')
    parser.add_argument('--initial_iters',
                        default=100,
                        type=int,
                        help='CoPur optimization steps to find the initial point')
    parser.add_argument('--final_iters',
                        default=20,
                        type=int,
                        help='CoPur optimization steps after the initial point is found')
    parser.add_argument('--tau',
                        default=100,
                        type=float,
                        help='CoPur loss scaling factor')
    
    args = parser.parse_args()
    return args
