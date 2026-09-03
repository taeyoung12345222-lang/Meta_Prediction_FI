import argparse
import util_v4 as util
import torch
from torch.utils.data import ConcatDataset
import torchvision
import torchvision.transforms as transforms
import os
import time
from my_datasets import CIFAR10, SVHN, SVHNBasic, CIFAR100

parser = argparse.ArgumentParser(description='CIFAR-10 baseline')
parser.add_argument('--model', 
                    default="QResNet8",
                    choices=['QResNet8'], 
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
parser.add_argument('--alpha',
                    default=0.1,
                    type=float,
                    help='alpha for dirichlet distribution')
parser.add_argument('--seed',
                    default=90,
                    type=int,
                    help='seed for random number generators')
parser.add_argument('--proxy_ratio',
                    default=0.1,
                    type=float,
                    help='ratio of proxy set to training set')
parser.add_argument('--batch_size',
                    default=16,
                    type=int,
                    help='batch size for logit generation')
parser.add_argument('--diff_init',
                    action='store_true',
                    help='whether to use different initializations for models, here for compatibility reasons')
parser.add_argument('--gpu',
                    action='store_true',
                    help='whether to use GPU')

args = parser.parse_args()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(model, data_loader):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

    return top1, top5

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def quantise(o_model, q_model, local_test_loader, test_loader, test=True):
    o_model = o_model.cpu().eval()
    q_model = q_model.eval()

    state_dict = o_model.state_dict()
    q_model.load_state_dict(state_dict)

    q_model.qconfig = torch.quantization.default_qconfig
    q_model.emb.qconfig = torch.quantization.float_qparams_weight_only_qconfig

    torch.quantization.prepare(q_model, inplace=True)

    print('==> Calibrating the model')
    # calibrate the model
    with torch.no_grad():
        for x, _ in local_test_loader:
            q_model(x)
    
    print('==> Quantising the model')
    torch.quantization.convert(q_model, inplace=True)

    if test:
        print('Size of model before quantization')
        print_size_of_model(o_model)

        print('Size of model after quantization')
        print_size_of_model(q_model)

        print('Computing test accuracy of the original model')
        top1, top5 = evaluate(o_model, test_loader)
        print(f'Local Test Set: Acc@1: {top1.avg:.3f}, Acc@5: {top5.avg:.3f}')

        print('Computing test accuracy of the quantised model')
        top1, top5 = evaluate(q_model, test_loader)
        print(f'Local Test Set: Acc@1: {top1.avg:.3f}, Acc@5: {top5.avg:.3f}')
        
    return q_model

class DummyArgs:
    def __init__(self, seed, datapath, alpha, dataset, proxy_ratio, batch_size):
        self.seed = seed
        self.datapath = datapath
        self.alpha=alpha
        self.dataset=dataset
        self.proxy_set=True
        self.proxy_ratio=proxy_ratio
        self.NIID=True
        self.tr_subset=False
        self.tr_subset_frac=1.0
        self.val_set=True
        self.val_ratio=0.5
        self.test_bs=batch_size
        self.bs=batch_size
        self.diff_init=False
        self.gpu=False
        self.apply_augmentation=False

if __name__ == '__main__':

    # check if save_dir and logits already exists
    if os.path.exists(args.save_dir):
        if os.path.exists(args.save_dir + '/logit_trainset.pkl') and os.path.exists(args.save_dir + '/logit_testset.pkl'):
            print('==> Logits already exist. Exiting...')
            exit(0)

    model = util.select_model(args)
    models, labels = util.load_models_and_label_dist(model, args.modelpath, map_location='cpu')
    
    dataset_args = DummyArgs(args.seed, args.datapath, args.alpha, args.dataset, args.proxy_ratio, args.batch_size)

    # load train and test sets if CIFAR10
    if args.dataset == 'CIFAR10':
        dataset = CIFAR10(len(models), dataset_args)
    elif args.dataset == 'CIFAR100':
        dataset = CIFAR100(len(models), dataset_args)  
    # with the extended training set
    elif args.dataset == 'SVHN':
        dataset = SVHN(len(models), dataset_args)    
    # without the extended training set
    elif args.dataset == 'SVHNBasic':
        dataset = SVHNBasic(len(models), dataset_args)
    else:
        raise ValueError(f"args.dataset set to {args.dataset}. Dataset not supported.")
    
    print("==> Beginning model quantisation")
    q_models = []
    for i, model_i in enumerate(models):
        print(f'Quantising model {i}')
        train_loader, proxy_loader, val_loader, test_loader, local_test_loader, num_samples = dataset.fetch(i)
        q_model = util.select_model(args, q=True)
        q_model = quantise(model_i, q_model, local_test_loader, test_loader, False)
        q_models.append(q_model)

    print("==> Generating trainset logits...")
    start = time.time()
    all_train_logits = {}

    for i in range(len(q_models)):
        print(f"==> Model {i}")
        train_loader, proxy_loader, val_loader, test_loader, local_test_loader, num_samples = dataset.fetch(i)
        logit_trainset = []
        with torch.no_grad():
            for elems, labels in proxy_loader:
                outputs = [model(elems).detach().cpu() for model in q_models]
                stacked_outputs = torch.hstack(outputs)
                logit_trainset.append((elems, stacked_outputs, labels))
        flattened_logit_trainset = []
        for x1, x2, y in logit_trainset:
            flattened_logit_trainset.extend(zip(x1, x2, y))
        all_train_logits[i] = flattened_logit_trainset

    end = time.time()
    gap_min = (end - start)//60
    gap_secs = (end - start) % 60
    print(f"==> Time taken to generate trainset logits: {gap_min} min {gap_secs:.2f} secs")
    
    print("==> Generating testset logits...")
    start = time.time()

    train_loader, proxy_loader, val_loader, test_loader, local_test_loader, num_samples = dataset.fetch(0)
    logit_testset = []
    with torch.no_grad():
        for elems, labels in test_loader:
            outputs = [model(elems).detach().cpu() for model in q_models]
            stacked_outputs = torch.hstack(outputs)
            logit_testset.append((elems, stacked_outputs, labels))
    flattened_logit_testset = []
    for x1, x2, y in logit_testset:
        flattened_logit_testset.extend(zip(x1, x2, y))
    all_test_logits = flattened_logit_testset

    end = time.time()
    gap_min = (end - start)//60
    gap_secs = (end - start) % 60
    print(f"==> Time taken to generate testset logits: {gap_min} min {gap_secs:.2f} secs")

    # save logits as a pickle file to args.save_dir
    torch.save(all_train_logits, args.save_dir + '/logit_trainset.pth', pickle_protocol=2)
    torch.save(all_test_logits, args.save_dir + '/logit_testset.pth', pickle_protocol=2)

    print('==> Done!')


