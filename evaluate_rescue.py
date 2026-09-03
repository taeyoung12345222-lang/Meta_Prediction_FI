"""
Stage 0 evaluation: for each aggregator, report Overall Accuracy (Acc_all) and
Rescue Accuracy (Acc_rescue) -- accuracy restricted to the subset of test samples
where the generalist majority is wrong but the relevant specialist is right.

Client ordering convention (must match the specialist_mix partition used for training):
client ids [0, n_generalists) are generalists, [n_generalists, n_generalists + len(specialist_classes))
are specialists, specialist i is the expert for specialist_classes[i].
"""
import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from Utils.general import get_model, get_num_classes
from aggregator_training_fl import generate_subsets_with_masks

parser = argparse.ArgumentParser(description='Stage 0 rescue accuracy evaluation')
parser.add_argument('--datapath', required=True, type=str, help='dir containing logit_testset.pth')
parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR100', 'AG_News'])
parser.add_argument('--models', required=True, type=str, nargs='+',
                     help='aggregator model names to evaluate, e.g. F_Avg F_Median2 F_Geo_Median F_TM2 DeepSet DeepSet_M DeepSet_TM')
parser.add_argument('--modelpaths', type=str, nargs='*', default=[],
                     help='for each learned model in --models (in the same relative order), the dir containing model.pth')
parser.add_argument('--n_generalists', required=True, type=int)
parser.add_argument('--specialist_classes', required=True, type=int, nargs='+')
parser.add_argument('--trim_ratio', default=0.25, type=float)
parser.add_argument('--dim_hidden', default=224, type=int)
parser.add_argument('--lip_scale', default=1.0, type=float)
parser.add_argument('--n_adv', default=4, type=int)
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--normalization_type', default='simplex', type=str,
                     choices=['range', 'fix-norm', 'simplex', 'simplex-one-hot'])
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--save_csv', default=None, type=str)
args = parser.parse_args()


def build_input(testset_raw, n_classes, normalize, normalization_type):
    testset = [(x.reshape(-1, n_classes), y) for x, y in testset_raw]
    if normalize:
        if normalization_type == 'simplex':
            testset = [(torch.softmax(x, dim=-1), y) for x, y in testset]
        elif normalization_type == 'simplex-one-hot':
            testset = [(torch.softmax(x, dim=-1), y) for x, y in testset]
            testset = [(F.one_hot(torch.argmax(x, dim=-1), num_classes=n_classes).float(), y) for x, y in testset]
        else:
            raise ValueError(f'Normalization type not supported for this script: {normalization_type}')
    return testset


def compute_rescue_mask(testset, n_generalists, specialist_classes):
    """ testset: list of (x [n_clients, n_classes] pre-normalization argmax-safe, y) """
    class_to_specialist = {c: i for i, c in enumerate(specialist_classes)}
    rescue_mask = np.zeros(len(testset), dtype=bool)
    applicable_mask = np.zeros(len(testset), dtype=bool)  # y in specialist_classes

    for idx, (x, y) in enumerate(testset):
        y = int(y)
        if y not in class_to_specialist:
            continue
        applicable_mask[idx] = True

        generalist_preds = torch.argmax(x[:n_generalists], dim=-1).numpy()
        vals, counts = np.unique(generalist_preds, return_counts=True)
        majority_pred = int(vals[np.argmax(counts)])

        si = class_to_specialist[y]
        specialist_pred = int(torch.argmax(x[n_generalists + si]).item())

        if majority_pred != y and specialist_pred == y:
            rescue_mask[idx] = True

    return rescue_mask, applicable_mask


def evaluate_model(model_name, modelpath, raw_testset, n_classes, n_clients, rng, device):
    output_prob = model_name in ['F_TM', 'F_Median', 'F_Median2'] and args.normalize and args.normalization_type == 'simplex'
    f = get_model(model_name, n_clients, n_classes, args.trim_ratio, args.dim_hidden, args.lip_scale, args.n_adv, output_prob)

    if f.state_dict() != {}:
        if modelpath is None:
            raise ValueError(f'{model_name} is a learned aggregator but no modelpath was given')
        f.load_state_dict(torch.load(os.path.join(modelpath, 'model.pth'), map_location='cpu'))
    f = f.to(device)
    f.eval()

    normed = build_input(raw_testset, n_classes, args.normalize, args.normalization_type)
    wrapped = generate_subsets_with_masks(normed, rng, n_clients, n_clients, n_clients, 1)
    loader = torch.utils.data.DataLoader(wrapped, batch_size=args.batch_size, shuffle=False, num_workers=2)

    y_preds, y_trues = [], []
    with torch.no_grad():
        for data, mask, target in loader:
            data, mask = data.to(device), mask.to(device)
            outputs = f(data, mask)
            y_preds.append(torch.argmax(outputs, dim=1).cpu())
            y_trues.append(target.cpu())

    return np.concatenate(y_preds), np.concatenate(y_trues)


if __name__ == '__main__':
    device = 'cuda' if args.gpu else 'cpu'
    n_classes = get_num_classes(args.dataset)
    rng = np.random.default_rng(seed=0)

    raw_testset_full = torch.load(os.path.join(args.datapath, 'logit_testset.pth'), map_location='cpu')
    raw_testset = [(x.reshape(-1, n_classes), y) for x, y in raw_testset_full]
    n_clients = raw_testset[0][0].shape[0]

    rescue_mask, applicable_mask = compute_rescue_mask(raw_testset, args.n_generalists, args.specialist_classes)
    print(f'==> {applicable_mask.sum()} / {len(raw_testset)} test samples have true label in specialist_classes')
    print(f'==> {rescue_mask.sum()} / {applicable_mask.sum()} of those are rescue cases '
          f'(generalist majority wrong, specialist right)')

    modelpaths = {}
    mp_iter = iter(args.modelpaths)
    # modelpaths are supplied positionally aligned to the models that need them; we match by
    # trying to load each learned model's path in the order given
    learned_models_in_order = []

    rows = []
    mp_idx = 0
    for model_name in args.models:
        # peek: static aggregators need no modelpath, learned ones consume the next --modelpaths entry
        output_prob = model_name in ['F_TM', 'F_Median', 'F_Median2'] and args.normalize and args.normalization_type == 'simplex'
        probe = get_model(model_name, n_clients, n_classes, args.trim_ratio, args.dim_hidden, args.lip_scale, args.n_adv, output_prob)
        needs_path = probe.state_dict() != {}
        modelpath = None
        if needs_path:
            if mp_idx >= len(args.modelpaths):
                raise ValueError(f'{model_name} needs a modelpath but --modelpaths ran out of entries')
            modelpath = args.modelpaths[mp_idx]
            mp_idx += 1

        y_preds, y_trues = evaluate_model(model_name, modelpath, raw_testset, n_classes, n_clients, rng, device)

        acc_all = accuracy_score(y_trues, y_preds)
        if rescue_mask.sum() > 0:
            acc_rescue = accuracy_score(y_trues[rescue_mask], y_preds[rescue_mask])
        else:
            acc_rescue = float('nan')

        print(f'{model_name:15s}  Acc_all={acc_all:.4f}  Acc_rescue={acc_rescue:.4f}  '
              f'(n_rescue={int(rescue_mask.sum())})')
        rows.append({
            'model': model_name,
            'Acc_all': acc_all,
            'Acc_rescue': acc_rescue,
            'n_rescue_samples': int(rescue_mask.sum()),
            'n_applicable_samples': int(applicable_mask.sum()),
        })

    df = pd.DataFrame(rows)
    print('\n' + df.to_string(index=False))
    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f'==> Saved results to {args.save_csv}')
