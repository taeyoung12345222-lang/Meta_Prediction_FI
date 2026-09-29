import numpy as np
import torch
import logging
from torch.utils.data import random_split
from math import floor

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]
    
    def __add__(self, other):
        combined_index = self.index + other.index
        return Partition(self.data, combined_index)
    
    def update_index(self, index):
        assert len(index) <= len(self.index)
        self.index = index

class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234, isNonIID=False, alpha=0, num_classes=10, \
                 dataset=None, proxyset=False, proxy_ratio=0.1, n_classes_per_client=2, partition_method='dirichlet', is_nlp=False, \
                 n_specialists=0, specialist_classes=None, specialist_purity=0.85, generalist_access_ratio=0.15, \
                 generalist_split='dirichlet'):
        self.data = data
        self.dataset = dataset
        self.num_classes = num_classes
        self.proxyset = proxyset
        self.proxy_ratio = proxy_ratio
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator().manual_seed(seed)

        if isNonIID:
            if partition_method == 'dirichlet':
                self.partitions, self.ratio = self.__getDirichletData__(data, sizes, self.rng, alpha, is_nlp=is_nlp)
            elif partition_method == 'sharding':
                self.partitions, self.ratio = self.__getShardedData__(data, sizes, self.rng, n_classes_per_client, is_nlp=is_nlp)
            elif partition_method == 'specialist_mix':
                self.partitions, self.ratio = self.__getSpecialistMixData__(data, sizes, self.rng, alpha, n_specialists, \
                                                specialist_classes, specialist_purity, generalist_access_ratio, is_nlp=is_nlp, \
                                                generalist_split=generalist_split)
        else:
            self.partitions = [] 
            data_len = len(data) 
            indexes = [x for x in range(0, data_len)] 
            self.rng.shuffle(indexes) 

            self.ratio = [] # counts instead of ratios
            for frac in sizes: 
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                self.ratio.append(part_len)
                indexes = indexes[part_len:]
            
            self.ratio = np.array(self.ratio)
        
        total_partitions = len(self.partitions)
        if self.proxyset:

            self.train_partitions = []; self.proxy_partitions = []

            # pick 10% randomly from each partition as proxy set
            for i in range(total_partitions):
                partition_len = len(self.partitions[i])
                proxy_len = int(partition_len * self.proxy_ratio)
                train_partition, proxy_partition = random_split(self.partitions[i], \
                                                                [partition_len - proxy_len, proxy_len], \
                                                                generator=self.generator)
                self.train_partitions.append(list(train_partition))
                self.proxy_partitions.append(list(proxy_partition))
                self.ratio[i] = partition_len - proxy_len # update to exclude proxy set

    def use(self, partition):
        if self.proxyset:
            return Partition(self.data, self.train_partitions[partition]), Partition(self.data, self.proxy_partitions[partition])
        return Partition(self.data, self.partitions[partition])

    def __getDirichletData__(self, data, psizes, rng, alpha, is_nlp=False):
        n_nets = len(psizes)
        K = self.num_classes

        labelList = []; idxs = [[] for _ in range(K)]
        if is_nlp:
            for i, d in enumerate(data):
                y = d['targets'].item()
                labelList.append(y)
                idxs[y].append(i)
        else:
            if isinstance(data, torch.utils.data.Subset):
                targets = [data.dataset.targets[idx] for idx in data.indices]
            else:
                targets = data.targets
            
            for i, y in enumerate(targets):
                labelList.append(y)
                idxs[y].append(i)

        labelList = np.array(labelList)
        idxs = [np.array(idxs[i]) for i in range(K)]

        min_size = 0
        N = len(labelList)

        net_dataidx_map = {}
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = idxs[k]
                rng.shuffle(idx_k)
                proportions = rng.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            rng.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            
        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
        logging.debug('Data statistics: %s' % str(net_cls_counts))

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        # weights = local_sizes/np.sum(local_sizes)
        weights = local_sizes # return counts insteads of ratios
        logging.info('Samples per client {}'.format(weights))

        return idx_batch, weights
    
    def __getSpecialistMixData__(self, data, psizes, rng, alpha, n_specialists, specialist_classes, \
                                  specialist_purity, generalist_access_ratio, is_nlp=False, generalist_split='dirichlet'):
        """ Splits clients into n_specialists 'specialist' clients (each dominated by one target
        class, drawn from a pool reserved ahead of time) plus generalist clients (the rest),
        who split what's left -- which is deliberately scarce on the specialist
        target classes, so the generalists have a genuine blind spot there instead of a random one.

        generalist_split='dirichlet' (default): generalists split the residual pool unevenly via
        Dirichlet(alpha), so a generalist can randomly end up over-represented in some class purely
        by chance (this is what earlier created "biased generalists" that mimic real specialists).
        generalist_split='equal': each class's residual is cut into n_generalists equal-sized chunks,
        so no generalist is randomly over- or under-exposed to any class relative to the others --
        isolates genuine specialization (from specialist_purity) from accidental data skew.

        Client ids [0, n_nets - n_specialists) are generalists; the last n_specialists ids are
        specialists, each assigned to specialist_classes[i] in order. """
        n_nets = len(psizes)
        n_generalists = n_nets - n_specialists

        assert n_specialists == len(specialist_classes), \
            f"n_specialists ({n_specialists}) must match len(specialist_classes) ({len(specialist_classes)})"
        assert len(set(specialist_classes)) == len(specialist_classes), \
            "specialist_classes must be distinct"
        assert n_generalists > 0, "need at least one generalist client"

        K = self.num_classes
        labelList = []; idxs = [[] for _ in range(K)]
        if is_nlp:
            for i, d in enumerate(data):
                y = d['targets'].item()
                labelList.append(y)
                idxs[y].append(i)
        else:
            if isinstance(data, torch.utils.data.Subset):
                targets = [data.dataset.targets[idx] for idx in data.indices]
            else:
                targets = data.targets
            for i, y in enumerate(targets):
                labelList.append(y)
                idxs[y].append(i)

        labelList = np.array(labelList)
        idxs = [np.array(idxs[k]) for k in range(K)]
        for k in range(K):
            rng.shuffle(idxs[k])

        N = len(labelList)
        target_size = N // n_nets  # per-client sample budget, matches the balanced dirichlet share

        # pool of indices still available for the generalist dirichlet split, per class
        class_pool = {k: idxs[k].copy() for k in range(K)}

        # carve out most of each target class for the specialist that owns it, leaving only a
        # scarce residual (generalist_access_ratio) for the generalist pool to dirichlet-split
        specialist_reserve = {}
        for c in specialist_classes:
            full = class_pool[c]
            n_residual = int(len(full) * generalist_access_ratio)
            class_pool[c] = full[:n_residual]
            specialist_reserve[c] = full[n_residual:]

        net_dataidx_map = {}
        specialist_ids = list(range(n_generalists, n_nets))

        for si, client_id in enumerate(specialist_ids):
            target_class = specialist_classes[si]
            pool = specialist_reserve[target_class]

            n_purity = min(int(target_size * specialist_purity), len(pool))
            purity_samples = pool[:n_purity]
            specialist_reserve[target_class] = pool[n_purity:]  # leftover reserve, unused

            n_impure_total = target_size - n_purity
            other_classes = [k for k in range(K) if k != target_class]
            per_class_impure = max(1, n_impure_total // len(other_classes))

            impure_chunks = []
            for oc in other_classes:
                take = min(per_class_impure, len(class_pool[oc]))
                impure_chunks.append(class_pool[oc][:take])
                class_pool[oc] = class_pool[oc][take:]

            client_idx = np.concatenate([purity_samples] + impure_chunks) if impure_chunks else purity_samples
            rng.shuffle(client_idx)
            net_dataidx_map[client_id] = client_idx.tolist()

        # split what's left (scarce on specialist_classes, ~full elsewhere) across the generalists
        if generalist_split == 'equal':
            idx_batch = [[] for _ in range(n_generalists)]
            for k in range(K):
                idx_k = class_pool[k].copy()
                rng.shuffle(idx_k)
                chunks = np.array_split(idx_k, n_generalists)  # near-equal sizes, no randomness in size
                idx_batch = [idx_j + chunk.tolist() for idx_j, chunk in zip(idx_batch, chunks)]
        else:
            # dirichlet-partition, following the same balancing logic as __getDirichletData__
            min_size = 0
            idx_batch = [[] for _ in range(n_generalists)]
            while min_size < K:
                idx_batch = [[] for _ in range(n_generalists)]
                for k in range(K):
                    idx_k = class_pool[k].copy()
                    rng.shuffle(idx_k)
                    proportions = rng.dirichlet(np.repeat(alpha, n_generalists))
                    proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                    if proportions.sum() == 0:
                        proportions = np.ones(n_generalists) / n_generalists
                    else:
                        proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_generalists):
            rng.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        partitions = [net_dataidx_map[j] for j in range(n_nets)]

        net_cls_counts = {}
        for net_i, dataidx in enumerate(partitions):
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            net_cls_counts[net_i] = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
        logging.info('Specialist client ids -> target class: %s', dict(zip(specialist_ids, specialist_classes)))
        logging.debug('Data statistics: %s' % str(net_cls_counts))

        local_sizes = np.array([len(p) for p in partitions])
        logging.info('Samples per client {}'.format(local_sizes))

        return partitions, local_sizes

    def __getShardedData__(self, data, psizes, rng, n_classes_per_client, is_nlp=False):
        n_nets = len(psizes)
        K = self.num_classes

        labelList = []; idxs = [[] for _ in range(K)]
        if is_nlp:
            for i, d in enumerate(data):
                y = d['targets'].item()
                labelList.append(y)
                idxs[y].append(i)
        else:
            for i, (_, y) in enumerate(data):
                labelList.append(y)
                idxs[y].append(i)

        labelList = np.array(labelList)
        idxs = [np.array(idxs[i]) for i in range(K)]

        total_shards = n_nets * n_classes_per_client
        n_shards_per_class = [floor(total_shards // K) for _ in range(K)]
        logging.info('Shards per class: {}'.format(n_shards_per_class))
        leftovers = total_shards - sum(n_shards_per_class)
        for i in range(leftovers):
            n_shards_per_class[i] += 1

        # split each class into n_shards_per_class shards after shuffling
        for i in range(K):
            rng.shuffle(idxs[i])
            idxs[i] = np.array_split(idxs[i], n_shards_per_class[i])
        
        # assign shards to clients
        net_dataidx_map = [[] for _ in range(n_nets)]
        classes_available = list(range(K))
        net_cls_counts = {i: {} for i in range(n_nets)}
        for i in range(n_nets):
            # choose n_classes_per_client class randomly
            for _ in range(n_classes_per_client):
                class_idx = rng.choice(classes_available)
                # choose the first shard of the class
                class_shard = idxs[class_idx].pop(0).tolist() 
                net_dataidx_map[i] += class_shard
                net_cls_counts[i][class_idx] = net_cls_counts[i].get(class_idx, 0) + len(class_shard)
                if len(idxs[class_idx]) == 0:
                    classes_available.remove(class_idx)
            
            logging.info('Client {} label distribution: {}'.format(i, net_cls_counts[i]))

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)

        logging.info('Samples per client {}'.format(local_sizes))

        return net_dataidx_map, local_sizes

def distribute_testset(test_set, label_dists, generator, is_nlp=False):
    num_clients = len(label_dists)
    num_classes = len(label_dists[0])
    
    # Calculate total counts per class across all clients
    total_counts = [sum(client_dists[c] for client_dists in label_dists) for c in range(num_classes)]

    # Calculate proportions per class per client
    proportions = [[client_dists[c] / total_counts[c] for c in range(num_classes)] for client_dists in label_dists]

    # Initialize partitions
    client_partitions = [Partition(test_set, []) for _ in range(num_clients)]

    test_label_dists = [[0] * num_classes for _ in range(num_clients)]

    # Sort and shuffle test set indices by class
    for c in range(num_classes):
        if is_nlp:
            class_indices_list = [i for i, item in enumerate(test_set) if item['targets'].item() == c]
        else:
            if isinstance(test_set, torch.utils.data.Subset):
                targets = [test_set.dataset.targets[idx] for idx in test_set.indices]
            else:
                targets = test_set.targets
            class_indices_list = [i for i, item in enumerate(targets) if item == c]
        shuffled_indices = torch.randperm(len(class_indices_list), generator=generator).tolist()
        class_indices_list = [class_indices_list[i] for i in shuffled_indices]
        
        # Calculate total and per-client item counts
        class_count = len(class_indices_list)
        per_client_counts = [int(proportions[client_id][c] * class_count) for client_id in range(num_clients)]
        distributed_count = sum(per_client_counts)
        
        # Adjust for rounding errors without exceeding class_count
        while distributed_count < class_count:
            for client_id in range(num_clients):
                if distributed_count >= class_count:
                    break
                if label_dists[client_id][c] > 0:  # Ensure non-zero proportion
                    per_client_counts[client_id] += 1
                    distributed_count += 1
        
        # Distribute items based on adjusted counts
        start_index = 0
        for client_id, num_items in enumerate(per_client_counts):
            end_index = start_index + num_items
            test_label_dists[client_id][c] = num_items
            client_partitions[client_id].index.extend(class_indices_list[start_index:end_index])
            start_index = end_index

    # print per_client_counts, label_dists for a randomly picked client
    random_client_id = torch.randint(0, num_clients, (1,)).item()
    logging.debug('Client {} label distribution: {}'.format(random_client_id, label_dists[random_client_id]))
    logging.debug('Client {} test distribution: {}'.format(random_client_id, test_label_dists[random_client_id]))
    logging.debug('Client {} test size: {}'.format(random_client_id, len(client_partitions[random_client_id])))
    total_test_set_size = sum([len(client_partitions[i]) for i in range(num_clients)])
    logging.debug('Total test set size of distributed clients: {}'.format(total_test_set_size))

    return client_partitions

def get_label_counts(train_loader, num_classes, is_nlp=False):
    labelList = []
    if is_nlp:
        for d in train_loader:
            labelList += d['targets'].tolist()
    else:
        for _, targets in train_loader:
            labelList += targets.tolist()
    
    # count labels per class
    label_counts = [0] * num_classes
    for label in labelList:
        label_counts[label] += 1
    
    return label_counts