from .Dataset import Dataset
from .DataPartitioner import DataPartitioner, distribute_testset, Partition

import torch
from torch.utils.data import random_split
import torchvision
from torchvision import transforms
import logging
import numpy as np

class CIFAR100(Dataset):

    def __init__(self, size, args):
        super().__init__(size, args)

        self.trainset = None
        self.proxyset = None
        self.valset = None
        self.testset = None
        self.label_dists = None
        self.local_testsets = None

        self.generator = torch.Generator().manual_seed(self.args.seed)
        self.num_classes = 100
        self.load_trainset()
        self.load_testset()

    def load_trainset(self):
        logging.info('==> load train data')
        if self.args.apply_augmentation:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        trainset = torchvision.datasets.CIFAR100(root=self.args.datapath, 
                                                train=True, 
                                                download=True, 
                                                transform=transform_train)
        
        

        # Take a subset of the combined dataset if specified
        if self.args.tr_subset:
            data_len = len(trainset)
            sub_len = int(data_len * self.args.tr_subset_frac)
            trainset, _ = random_split(trainset, [sub_len, data_len - sub_len], generator=self.generator)
        self.trainset = trainset

        partition_sizes = [1.0 / self.size for _ in range(self.size)]
        self.partition = DataPartitioner(self.trainset, partition_sizes, isNonIID=self.args.NIID, num_classes=self.num_classes, \
                                        seed=self.args.seed, alpha=self.args.alpha, dataset=self.args.dataset, \
                                        proxyset=self.args.proxy_set, proxy_ratio=self.args.proxy_ratio, \
                                        n_classes_per_client=self.args.n_classes_per_client, partition_method=self.args.partition_method)
        self.num_samples = self.partition.ratio # returns counts intead of ratios

        logging.info('==> Finished partitioning data')
        
        # Build a single global proxyset
        if self.args.proxy_set:
            logging.info('==> Collecting proxy sets from all clients')
            all_proxy_indices = []
            for i in range(self.size):
                _, proxy_part = self.partition.use(i)
                all_proxy_indices.extend(proxy_part.index)  # just collect indices
            self.proxyset = Partition(self.trainset, all_proxy_indices)
    
    def load_testset(self):
        logging.info('==> load test data')
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        testset = torchvision.datasets.CIFAR100(root=self.args.datapath, 
                                            train=False, 
                                            download=True, 
                                            transform=transform_test)

        if self.args.val_set:
            val_len = int(len(testset) * self.args.val_ratio)
            self.valset, self.testset = random_split(testset, [val_len, len(testset) - val_len], generator=self.generator)
        else:
            self.testset = testset
        
        logging.info('==> val set size: {}'.format(len(self.valset) if self.valset else 0))
        logging.info('==> test set size: {}'.format(len(self.testset)))

    def fetch(self, client_index):
        proxy_loader = None
        if self.args.proxy_set:
            logging.info(f'Creating proxy loader')
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
        
        logging.info(f'Creating train loader')
        train_loader = torch.utils.data.DataLoader(client_data, 
                                            batch_size=self.args.bs, 
                                            shuffle=True, 
                                            num_workers=2)
        
        logging.info('Creating test loader')
        test_loader = torch.utils.data.DataLoader(self.testset, 
                                            batch_size=self.args.test_bs, 
                                            shuffle=False, 
                                            num_workers=2)
        val_loader = None
        if self.valset:
            logging.info('Creating val loader')
            val_loader = torch.utils.data.DataLoader(self.valset, 
                                            batch_size=self.args.test_bs, 
                                            shuffle=False, 
                                            num_workers=2)
        
        logging.info('Creating local test loader')
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
            # access original targets
            if isinstance(self.trainset, torch.utils.data.Subset):
                base_targets = [self.trainset.dataset.targets[idx] for idx in self.trainset.indices]
            else:
                base_targets = self.trainset.targets
            for i in range(self.size):
                logging.info(f'Computing label distribution for client {i}')
                if self.args.proxy_set:
                    client_data, _ = self.partition.use(i)
                else:
                    client_data = self.partition.use(i)

                indices = client_data.index
                labels = np.array(base_targets)[indices]
                label_dist_i = np.bincount(labels, minlength=self.num_classes).tolist()
                logging.info(f'Label distribution for client {i}: {label_dist_i}')
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
        