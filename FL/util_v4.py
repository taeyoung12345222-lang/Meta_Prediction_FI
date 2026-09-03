import torch
from my_datasets import *
from models import *
import pickle, os, copy

def get_dataset(size, dataset, args):
    datasets = {
        'CIFAR10': CIFAR10,
        'SVHN': SVHN,
        'SVHNBasic': SVHNBasic,
        'CIFAR100': CIFAR100,
        'MNIST': MNIST,
        'AG_News': AG_News
    }
    return datasets[dataset](size, args)

def get_num_classes(dataset):
    num_classes = {
        'CIFAR10': 10,
        'SVHN': 10,
        'SVHNBasic': 10,
        'CIFAR100': 100,
        'MNIST': 10,
        'AG_News': 4
    }
    return num_classes[dataset]

def select_model(args, rank=0, q=False):
    if args.diff_init:
        torch.manual_seed(rank+args.seed)
        if args.gpu:
            torch.cuda.manual_seed(rank+args.seed)
    else:
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed(args.seed)
    if args.model == 'VGG':
        model = vgg11()
    elif args.model == 'CNN':
        model = CNN()
    elif args.model == 'ResNet8':
        model = ResNet8(num_classes=get_num_classes(args.dataset))
    elif args.model == 'SmallNN':
        model = SmallNN(d=args.d, total_clients=args.totalclients, num_classes=get_num_classes(args.dataset))
    elif args.model == 'WeightedNN':
        model = WeightedAggregatorNN(total_clients=args.totalclients, num_classes=get_num_classes(args.dataset))
    elif args.model == 'LinearAggregator':
        model = LinearAggregator(total_clients=args.totalclients, num_classes=get_num_classes(args.dataset))
    elif args.model == 'QResNet8':
        model = QResNet8(num_classes=get_num_classes(args.dataset), q=q)
    elif args.model == 'FederatedMoE':
        model = FederatedMoE(num_experts=args.totalclients)
    elif args.model == 'FederatedMoE2':
        model = FederatedMoE2(num_experts=args.totalclients, num_classes=get_num_classes(args.dataset))
    elif args.model == 'DistilBert':
        model = DistilBERT(num_classes=get_num_classes(args.dataset))
    elif args.model == 'ViT_B32':
        model = ViT_B32(num_classes=get_num_classes(args.dataset))

    # You can add more models here
    return model

def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_models_and_label_dist(my_model, save_dir, map_location=None, use_best=False):
    all_files = os.listdir(save_dir)
    # load models with format "{rank}_model.pth"
    if not use_best:
        model_files = [f for f in all_files if f.endswith('_model.pth') and not f.endswith('_best_model.pth')]
    else:
        model_files = [f for f in all_files if f.endswith('_best_model.pth')]
    model_files.sort(key=lambda x: int(x.split('_')[0]))
    models = []
    for f in model_files:
        if map_location:
            loaded_state = torch.load(os.path.join(save_dir, f), map_location=map_location)
        else:
            loaded_state = torch.load(os.path.join(save_dir, f))
        my_model.load_state_dict(loaded_state)
        models.append(copy.deepcopy(my_model))

    # load labels with format "{rank}_label.pk"
    label_files = [f for f in all_files if f.endswith('_label.pk') or f.endswith('_label_distribution.pk')]
    label_files.sort(key=lambda x: int(x.split('_')[0]))
    labels = []
    for f in label_files:
        with open(os.path.join(save_dir, f), 'rb') as f:
            labels.append(pickle.load(f))
    
    print(f'==> Total models loaded {len(models)}')
    return models, labels

class Meter(object):
    """ Computes and stores the average, variance, and current value """

    def __init__(self, init_dict=None, ptag='Time', stateful=False,
                 csv_format=True):
        """
        :param init_dict: Dictionary to initialize meter values
        :param ptag: Print tag used in __str__() to identify meter
        :param stateful: Whether to store value history and compute MAD
        """
        self.reset()
        self.ptag = ptag
        self.value_history = None
        self.stateful = stateful
        if self.stateful:
            self.value_history = []
        self.csv_format = csv_format
        if init_dict is not None:
            for key in init_dict:
                try:
                    # TODO: add type checking to init_dict values
                    self.__dict__[key] = init_dict[key]
                except Exception:
                    print('(Warning) Invalid key {} in init_dict'.format(key))

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        self.sqsum = 0
        self.mad = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sqsum += (val ** 2) * n
        if self.count > 1:
            self.std = ((self.sqsum - (self.sum ** 2) / self.count)
                        / (self.count - 1)
                        ) ** 0.5
        if self.stateful:
            self.value_history.append(val)
            mad = 0
            for v in self.value_history:
                mad += abs(v - self.avg)
            self.mad = mad / len(self.value_history)

    def __str__(self):
        if self.csv_format:
            if self.stateful:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.mad:.3f}'
                           .format(dm=self))
            else:
                return str('{dm.val:.3f},{dm.avg:.3f},{dm.std:.3f}'
                           .format(dm=self))
        else:
            if self.stateful:
                return str(self.ptag) + \
                       str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.mad:.3f})'
                           .format(dm=self))
            else:
                return str(self.ptag) + \
                       str(': {dm.val:.3f} ({dm.avg:.3f} +- {dm.std:.3f})'
                           .format(dm=self))

