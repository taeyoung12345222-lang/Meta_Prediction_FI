import torch
import os

@torch.no_grad()
def distributed_feature_flipping(data_loader, amplification, n_adv, n_clients, n_classes, shuffle_order):
    if n_adv == 0:
        return data_loader
 
    new_data = []
    nadv_indices = shuffle_order[-n_adv:]

    for x, y in data_loader:
        x = x.reshape(-1, n_clients, n_classes)
        x[:, nadv_indices] = -amplification * x[:, nadv_indices]

        new_data.append((x.reshape(-1, n_clients * n_classes), y))
    
    return new_data

@torch.no_grad()
def distributed_feature_flipping_with_data_loading(datapath, shuffle_order, amplification, 
                                                   n_adv, n_clients, n_classes):
    testset = torch.load(os.path.join(datapath, 'logit_testset.pth'), map_location='cpu')
    if n_adv == 0:
        return testset
 
    new_data = []
    for x, y in testset:
        
        x = x.reshape(n_clients, n_classes)
        x = x[shuffle_order]
        x[-n_adv:] = -amplification * x[-n_adv:]

        new_data.append((x, y))
        
    return new_data