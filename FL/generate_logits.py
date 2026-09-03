import argparse
import util_v4 as util
import torch
from torch.utils.data import ConcatDataset
import torchvision
import torchvision.transforms as transforms
import os
import time
    
parser = argparse.ArgumentParser(description='CIFAR-10 baseline')
parser.add_argument('--model', 
                    default="ResNet8",
                    choices=['ResNet8', 'VGG'], 
                    type=str, 
                    help='neural network model')
parser.add_argument('--datapath',
                    required=True,
                    type=str,
                    help='directory from where to load datasets')
parser.add_argument('--modelpath',
                    required=True,
                    type=str,
                    help='directory from where to load trained model for aggregator training')
parser.add_argument('--save_dir',
                    required=True,
                    type=str,
                    help='directory to save logits')
parser.add_argument('--dataset',
                    default='CIFAR10',
                    type=str,
                    choices=['CIFAR10', 'SVHN', 'SVHNBasic', 'CIFAR100'],
                    help='which dataset to run on')
parser.add_argument('--gpu', '-g', 
                    action='store_true', # default value is False
                    help='whether to run on gpu')
parser.add_argument('--diff_init',
                    action='store_true',
                    help='whether to use different initializations for models, here for compatibility reasons')
parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='seed for random number generators, here for compatibility reasons')

args = parser.parse_args()


if __name__ == '__main__':

    # check if save_dir and logits already exists
    if os.path.exists(args.save_dir):
        if os.path.exists(args.save_dir + '/logit_trainset.pkl') and os.path.exists(args.save_dir + '/logit_testset.pkl'):
            print('==> Logits already exist. Exiting...')
            exit(0)

    if(args.gpu and not torch.cuda.is_available()):
        raise ValueError(f"args.gpu set to {args.gpu}. GPU not available/detected on this machine.")

    model = util.select_model(args)
    models, labels = util.load_models_and_label_dist(model, args.modelpath, map_location='cpu')
    if args.gpu:
        models = [model.cuda() for model in models]
    
    # load train and test sets if CIFAR10
    if args.dataset == 'CIFAR10':
        
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=args.datapath, 
                                                train=True, 
                                                download=True, 
                                                transform=transform_train)

        testset = torchvision.datasets.CIFAR10(root=args.datapath, 
                                                train=False, 
                                                download=True, 
                                                transform=transform_train)
    elif args.dataset == 'CIFAR100':
            
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = torchvision.datasets.CIFAR100(root=args.datapath, 
                                                train=True, 
                                                download=True, 
                                                transform=transform_train)

        testset = torchvision.datasets.CIFAR100(root=args.datapath, 
                                                train=False, 
                                                download=True, 
                                                transform=transform_train)
    
    # with the extended training set
    elif args.dataset == 'SVHN':
        
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = torchvision.datasets.SVHN(root=args.datapath, 
                                                split='train', 
                                                download=True, 
                                                transform=transform_train)
        extraset = torchvision.datasets.SVHN(root=args.datapath, 
                                                split='extra', 
                                                download=True, 
                                                transform=transform_train)
        trainset = ConcatDataset([trainset, extraset])
        
        testset = torchvision.datasets.SVHN(root=args.datapath, 
                                            split='test', 
                                            download=True, 
                                            transform=transform_train)
    
    # without the extended training set
    elif args.dataset == 'SVHNBasic':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = torchvision.datasets.SVHN(root=args.datapath, 
                                                split='train', 
                                                download=True, 
                                                transform=transform_train)        
        testset = torchvision.datasets.SVHN(root=args.datapath, 
                                            split='test', 
                                            download=True, 
                                            transform=transform_train)
    else:
        raise ValueError(f"args.dataset set to {args.dataset}. Dataset not supported.")
    
    # create dataloaders
    train_loader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=512, 
                                                shuffle=False, 
                                                num_workers=4)
    
    test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=512,
                                                shuffle=False,
                                                num_workers=4)
    
    
    print("==> Generating trainset logits...")
    
    start = time.time()
    
    logit_trainset = []
    with torch.no_grad():
        for elems, labels in train_loader:
            elems = elems.cuda() if(args.gpu) else elems
            outputs = [model(elems).detach().cpu() for model in models]
            stacked_outputs = torch.hstack(outputs)
            logit_trainset.append((stacked_outputs, labels))
    flattened_logit_trainset = []
    for x, y in logit_trainset:
        flattened_logit_trainset.extend(zip(x, y))

    end = time.time()
    gap_min = (end - start)//60
    gap_secs = (end - start) % 60
    print(f"==> Time taken to generate trainset logits: {gap_min} min {gap_secs:.2f} secs")
    print("==> Generating testset logits...")

    start = time.time()
    logit_testset = []
    with torch.no_grad():
        for elems, labels in test_loader:
            elems = elems.cuda() if(args.gpu) else elems
            outputs = [model(elems).detach().cpu() for model in models]
            stacked_outputs = torch.hstack(outputs)
            logit_testset.append((stacked_outputs, labels))
    flattened_logit_testset = []
    for x, y in logit_testset:
        flattened_logit_testset.extend(zip(x, y))

    end = time.time()
    gap_min = (end - start)//60
    gap_secs = (end - start) % 60
    print(f"==> Time taken to generate testset logits: {gap_min} min {gap_secs:.2f} secs")
    print('==> Saving logits...')

    # save logits as a pickle file to args.save_dir
    torch.save(flattened_logit_trainset, args.save_dir + '/logit_trainset.pth', pickle_protocol=2)
    torch.save(flattened_logit_testset, args.save_dir + '/logit_testset.pth', pickle_protocol=2)

    print('==> Done!')


