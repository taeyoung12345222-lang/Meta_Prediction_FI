from .Dataset import Dataset
from .DataPartitioner import DataPartitioner, get_label_counts, distribute_testset

import torch
from torch.utils.data import random_split
import torchvision
from torchvision import transforms
import logging

class SVHNBasic(Dataset):

    def __init__(self, size, args):
        super().__init__(size, args)
        self.trainset = None
        self.proxyset = None
        self.valset = None
        self.testset = None
        self.label_dists = None
        self.local_testsets = None

        self.generator = torch.Generator().manual_seed(self.args.seed)
        self.num_classes = 10
        self.load_trainset()
        self.load_testset()
        
        
    def load_trainset(self):
        logging.info('==> load train data')
        if self.args.apply_augmentation:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        trainset = torchvision.datasets.SVHN(root=self.args.datapath, 
                                                split='train', 
                                                download=True, 
                                                transform=transform_train)

        # Take a subset of the combined dataset if specified
        if self.args.tr_subset:
            # split using random_split
            data_len = len(trainset)
            sub_len = int(data_len * self.args.tr_subset_frac)
            trainset, _ = \
                random_split(trainset, [sub_len, data_len - sub_len], generator=self.generator)
        self.trainset = trainset
        
        logging.info('==> train set size: {}'.format(len(self.trainset)))

        partition_sizes = [1.0 / self.size for _ in range(self.size)]
        self.partition = DataPartitioner(self.trainset, partition_sizes, isNonIID=self.args.NIID, num_classes=self.num_classes, \
                                        seed=self.args.seed, alpha=self.args.alpha, dataset=self.args.dataset, \
                                        proxyset=self.args.proxy_set, proxy_ratio=self.args.proxy_ratio, \
                                        n_classes_per_client=self.args.n_classes_per_client, partition_method=self.args.partition_method)
        self.num_samples = self.partition.ratio # returns counts intead of ratios

        self.proxyset = []
        if self.args.proxy_set: # global proxyset
            for i in range(self.size):
                self.proxyset.extend(self.partition.use(i)[1])

    def load_testset(self):
        logging.info('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        testset = torchvision.datasets.SVHN(root=self.args.datapath, 
                                            split='test', 
                                            download=True, 
                                            transform=transform_test)

        if self.args.val_set:
            val_len = int(len(testset) * self.args.val_ratio)
            self.valset, self.testset = \
                random_split(testset, [val_len, len(testset) - val_len], generator=self.generator)
        else:
            self.testset = testset
        
        logging.info('==> test set size: {}'.format(len(self.testset)))
        logging.info('==> val set size: {}'.format(len(self.valset) if self.valset else 0))

    def fetch(self, client_index):
        proxy_loader = None
        if self.args.proxy_set:
            client_data, proxy_data = self.partition.use(client_index)

            proxy_loader = torch.utils.data.DataLoader(proxy_data, 
                                            batch_size=self.args.test_bs, 
                                            shuffle=True, 
                                            num_workers=2)
            logging.info('==> Client id {}, train dataset size {}, proxy set size: {}' \
                          .format(client_index, len(client_data), len(proxy_data)))
        
        else:
            client_data = self.partition.use(client_index)
            logging.info('==> Client id {}, train dataset size {}, proxy set size: {}' \
                          .format(client_index, len(client_data), 0))
            
        train_loader = torch.utils.data.DataLoader(client_data, 
                                            batch_size=self.args.bs, 
                                            shuffle=True, 
                                            pin_memory=True)
        test_loader = torch.utils.data.DataLoader(self.testset, 
                                            batch_size=self.args.test_bs, 
                                            shuffle=False, 
                                            num_workers=2)
        val_loader = None
        if self.valset:
            val_loader = torch.utils.data.DataLoader(self.valset, 
                                            batch_size=self.args.test_bs, 
                                            shuffle=False, 
                                            num_workers=2)
        
        local_testset = self.get_local_testset(client_index)
        local_test_loader = torch.utils.data.DataLoader(local_testset, 
                                            batch_size=self.args.test_bs, 
                                            shuffle=False, 
                                            num_workers=1)
        
        return train_loader, proxy_loader, val_loader, test_loader, local_test_loader, self.num_samples

    # combination of all local proxy sets
    def get_proxyset(self):
        if self.args.proxy_set:
            proxy_loader = torch.utils.data.DataLoader(self.proxyset, 
                                            batch_size=self.args.test_bs, 
                                            shuffle=True, 
                                            num_workers=2)
            return proxy_loader
        else:
            raise ValueError('Proxy set not available')

    def get_label_dist(self, client_index):
        if not self.label_dists:
            label_dists = []
            for i in range(self.size):
                if self.args.proxy_set:
                    client_data, _ = self.partition.use(i)
                else:
                    client_data = self.partition.use(i)
                local_train_loader = torch.utils.data.DataLoader(client_data, 
                                                    batch_size=self.args.test_bs, 
                                                    shuffle=False, 
                                                    num_workers=1)
                label_dist_i = get_label_counts(local_train_loader, self.num_classes)
                label_dists.append(label_dist_i)
            self.label_dists = label_dists

        return self.label_dists[client_index]

    def get_local_testset(self, client_index):
        
        # computes self.label_dists if not already computed
        if self.label_dists is None:
            self.get_label_dist(client_index)

        # computes self.local_testsets if not already computed
        if self.local_testsets is None:
            self.local_testsets = distribute_testset(self.testset, self.label_dists, self.generator)
        
        return self.local_testsets[client_index]