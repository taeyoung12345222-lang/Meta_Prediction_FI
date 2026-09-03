import argparse

def get_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 baseline')
    parser.add_argument('--size',
                        default=20,
                        type=int,
                        help='number of clients')
    parser.add_argument('--model', 
                        default="SmallNN",
                        choices=['SmallNN', 'DeepSet', 'DeepSet_TM', \
                                 'F_Avg', 'F_Median', 'F_Geo_Median', 'F_TM',  
                                 'DeepSet_M', 'DeepSet_Median', 'F_Median2', 
                                 'F_TM2', 'DeepSet_TM2', 'DeepSet_Median2'],
                        type=str, 
                        help='neural network model')
    parser.add_argument('--datapath',
                        required=True,
                        type=str,
                        help='directory where the generated logits are stored')
    parser.add_argument('--modelpath',
                        required=False,
                        type=str,
                        default=None,
                        help='directory where the trained model is stored [used during testing]')
    parser.add_argument('--save_dir',
                        required=True,
                        type=str,
                        help='directory to save outputs')
    parser.add_argument('--dataset',
                        default='CIFAR10',
                        type=str,
                        choices=['CIFAR10', 'CIFAR100', 'AG_News'],
                        help='which dataset to run on')
    parser.add_argument('--alpha',
                        default=0.1,
                        type=float,
                        help='alpha for dirichlet distribution')
    parser.add_argument('--n_classes_per_client',
                        default=4,
                        type=int,
                        help='number of classes per client for sharding based heterogeneity')
    parser.add_argument('--partition_method',
                        default='dirichlet',
                        choices=['dirichlet', 'sharding'],
                        type=str,
                        help='how to partition the dataset')
    parser.add_argument('--seed',
                        default=90,
                        type=int,
                        help='seed for random number generators')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='batch size for logit generation')
    parser.add_argument('--gpu',
                        action='store_true',
                        help='whether to use GPU')
    
    parser.add_argument('--S_path',
                        default="",
                        type=str,
                        help='path to similarity matrix for CPA attack')
    
    # DeepSet training parameters
    parser.add_argument('--add_subsets',
                        action='store_true',
                        help='whether to use GPU')
    parser.add_argument('--max_set_size',
                        default=10,
                        type=int,
                        help='maximum number of clients to consider while training')
    parser.add_argument('--min_set_size',
                        default=5,
                        type=int,
                        help='minimum number of clients to consider while training')
    parser.add_argument('--n_subsets',
                        default=5,
                        type=int,
                        help='number of subsets to create for every input set')
    parser.add_argument('--dim_hidden',
                        default=128,
                        type=int,
                        help='dimension of hidden layers in DeepSet')
    
    # Adversarial attack and defense parameters
    parser.add_argument('--adversarial',
                        action='store_true',
                        help='whether to use adversarial training [or testing]')
    parser.add_argument('--n_adv',
                        default=2,
                        type=int,
                        help='number of adversaries')
    parser.add_argument('--eval_one_adv',
                        action='store_true',
                        help='whether to evaluate only for n_adv adversaries and not all values from [1, n_adv]')
    parser.add_argument('--n_iter',
                        default=50,
                        type=int,
                        help='number of training iterations for adversarial attack')
    parser.add_argument('--normalize',
                        action='store_true',
                        help='whether to normalize logits')
    parser.add_argument('--normalization_type',
                        default='simplex',
                        choices=['range', 'fix-norm', 'simplex', 'simplex-one-hot'],
                        type=str,
                        help='type of normalization')
    parser.add_argument('--norm_value',
                        default=1.0,
                        type=float,
                        help='norm value for fix-norm | [-norm_value, norm_value] for range')
    parser.add_argument('--attack_type',
                        default='pgd',
                        choices=['sia', 'pgd', 'dfl', 'lma', 'cpa', 'ia'],
                        type=str,
                        help='type of attack')
    parser.add_argument('--loss_fn',
                        default='ce',
                        choices=['ce', 'cw'],
                        type=str,
                        help='choice of loss for adversarial testing with PGD - cross-entropy or Carlini-Wagner')
    parser.add_argument('--cw_confidence',
                        default=0.1,
                        type=float,
                        help='confidence margin for Carlini-Wagner loss')
    parser.add_argument('--amplification',
                        type=float,
                        default=1,
                        help='amplification factor for dfl attack')

    # Randomized Ablation defense
    parser.add_argument('--count',
                        default=100,
                        type=int,
                        help='number of random ablations')

    # Lipschitz defense
    parser.add_argument('--lip_scale',
                        default=1.0,
                        type=float,
                        help='Lipschitz scaling factor')

    # trimmed mean defense
    parser.add_argument('--trim_ratio',
                        default=0.1,
                        type=float,
                        help='trimming ratio')
    
    # logging
    parser.add_argument('--debug',
                        action='store_true',
                        help='whether to log debug messages')
    
    # Attack testing
    parser.add_argument('--new_adversaries',
                        action='store_true',
                        help='if true, adveraries are different from clients | if false, adversaries are a subset of clients')
    parser.add_argument('--black_box',
                        action='store_true',
                        help='if true, adversaries are black-box | if false, adversaries are white-box')
    parser.add_argument('--collude',
                        action='store_true',
                        help='if true, adversaries collude with each other | this is always true in white-box attacks')
    parser.add_argument('--save_attacked_logits',
                        action='store_true',
                        help='if true, save attacked logits to disk')

    # FL training
    parser.add_argument('--proxy_ratio',
                        default=0.1,
                        type=float,
                        help='ratio of proxy set to training set')

    # training params
    parser.add_argument('--lr',
                        default=0.1,
                        type=float,
                        help='learning rate')
    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        help='number of epochs')
    parser.add_argument('--optimizer',
                        default='SGD',
                        choices=['SGD', 'Adam'],
                        type=str,
                        help='optimizer')

    # wandb args
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

    args = parser.parse_args()
    return args