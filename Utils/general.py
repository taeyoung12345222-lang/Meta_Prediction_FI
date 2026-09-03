from Models.mlp import SmallNN
from Models.deepset import DeepSet, DeepSet_Median2, DeepSet_TM, \
                            DeepSet_M, DeepSet_Median, DeepSet_TM2
from Models.static_aggs import F_Avg, F_Median, F_Geo_Median, F_TM, F_Median2, F_TM2


def get_model(model_name, n_clients, n_classes, trim_ratio, dim_hidden, lip_scale=0.5, n_adv=2, output_prob=False):
    if model_name == 'SmallNN':
        f = SmallNN(n_clients, n_classes)
    elif model_name == 'DeepSet':
        print(f'==> Using DeepSet aggregation, output_prob={output_prob}')
        f = DeepSet(n_classes, n_classes, output_prob=output_prob, dim_hidden=dim_hidden)
    elif model_name == 'DeepSet_Median':
        print(f'==> Using Median aggregation, output_prob={output_prob}')
        f = DeepSet_Median(n_classes, n_classes, dim_hidden=dim_hidden)
    elif model_name == 'DeepSet_Median2':
        print(f'==> Using Median2 aggregation, output_prob={output_prob}')
        f = DeepSet_Median2(n_classes, n_classes, dim_hidden=dim_hidden)
    elif model_name == 'DeepSet_TM':
        print(f'==> Using DeepSet_TM aggregation, trim_ratio={trim_ratio}, output_prob={output_prob}')
        f = DeepSet_TM(n_classes, n_classes, trim_ratio=trim_ratio, output_prob=output_prob, dim_hidden=dim_hidden)
    elif model_name == 'DeepSet_TM2':
        print(f'==> Using DeepSet_TM2 aggregation, trim_ratio={trim_ratio}, output_prob={output_prob}')
        f = DeepSet_TM2(n_classes, n_classes, trim_ratio=trim_ratio, output_prob=output_prob, dim_hidden=dim_hidden)
    elif model_name == 'DeepSet_M':
        f = DeepSet_M(n_classes, n_classes, dim_hidden=dim_hidden)
    elif model_name == 'F_Avg':
        f = F_Avg(n_classes)
    elif model_name == 'F_Median':
        print(f'==> Using Median aggregation, output_prob={output_prob}')
        f = F_Median(n_classes, output_prob)
    elif model_name == 'F_Median2':
        print(f'==> Using Median2 aggregation, output_prob={output_prob}')
        f = F_Median2(n_classes, output_prob)
    elif model_name == 'F_Geo_Median':
        print(f'==> Using Geometric Median aggregation, output_prob={output_prob}')
        f = F_Geo_Median(n_classes, output_prob)
    elif model_name == 'F_TM':
        print(f'==> Using TM aggregation, trim_ratio={trim_ratio}, output_prob={output_prob}')
        f = F_TM(n_classes, trim_ratio, output_prob)
    elif model_name == 'F_TM2':
        print(f'==> Using TM2 aggregation, trim_ratio={trim_ratio}, output_prob={output_prob}')
        f = F_TM2(n_classes, trim_ratio, output_prob)
    else:
        raise ValueError(f'Model not supported {model_name}')
    
    return f

def get_num_classes(dataset):
    num_classes = {
        'CIFAR10': 10,
        'CIFAR100': 100,
        'AG_News': 4
    }
    return num_classes[dataset]