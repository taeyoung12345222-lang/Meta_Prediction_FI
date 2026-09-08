"""
Stage 1 evaluation: does historical behavior q_i(c) actually carry specialization
information about client i on class c?

q_i(c) = (1/T) * sum_t p_i(c | x_t)          -- algorithm's observation, no GT used
E_i(c) = Acc_i(c)                             -- ground-truth per-class accuracy, eval only
BiasScore_i(c) = q_i(c) - E_i(c)

q_i is computed from the proxy set (logit_trainset.pth): every client's own softmax
confidence in class c, averaged over the shared proxy pool -- this is exactly the
"historical behavior" signal available to the system without ground truth.
E_i is computed from the held-out test set (logit_testset.pth), which does carry GT,
and is used only to check whether q_i is a valid proxy for real expertise.
"""
import argparse
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from scipy.stats import spearmanr

from Utils.general import get_num_classes

parser = argparse.ArgumentParser(description='Stage 1: does q_i carry specialization info?')
parser.add_argument('--datapath', required=True, type=str, help='dir containing logit_trainset.pth / logit_testset.pth')
parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR100', 'AG_News'])
parser.add_argument('--n_generalists', required=True, type=int)
parser.add_argument('--specialist_classes', required=True, type=int, nargs='+')
parser.add_argument('--top_k', default=3, type=int, help='top-k for specialist ranking eval')
parser.add_argument('--bias_threshold', default=0.1, type=float, help='BiasScore above this counts as "large positive"')
parser.add_argument('--save_csv', default=None, type=str)
args = parser.parse_args()


def load_flat(path, n_classes):
    raw = torch.load(path, map_location='cpu')
    if isinstance(raw, dict):
        flat = []
        for i in sorted(raw.keys()):
            flat += raw[i]
    else:
        flat = raw
    xs = torch.stack([x.reshape(-1, n_classes) for x, y in flat]).float()  # [T, n_clients, n_classes]
    ys = torch.tensor([int(y) for _, y in flat])
    return xs, ys  # xs: [T, n_clients, n_classes], ys: [T]


if __name__ == '__main__':
    n_classes = get_num_classes(args.dataset)
    n_specialists = len(args.specialist_classes)
    n_clients = args.n_generalists + n_specialists
    specialist_ids = list(range(args.n_generalists, n_clients))
    role = {i: ('specialist', args.specialist_classes[i - args.n_generalists]) if i in specialist_ids else ('generalist', None)
            for i in range(n_clients)}

    X_proxy, _ = load_flat(os.path.join(args.datapath, 'logit_trainset.pth'), n_classes)
    X_test, y_test = load_flat(os.path.join(args.datapath, 'logit_testset.pth'), n_classes)
    print(f'==> proxy set: {X_proxy.shape[0]} samples, test set: {X_test.shape[0]} samples, {n_clients} clients')

    # q_i(c): mean predicted probability for class c over the proxy pool, per client
    P_proxy = F.softmax(X_proxy, dim=-1)          # [T, n_clients, n_classes]
    Q = P_proxy.mean(dim=0).numpy()                # [n_clients, n_classes]

    # E_i(c): client i's own accuracy restricted to test samples whose true label is c
    preds_test = torch.argmax(X_test, dim=-1)      # [N, n_clients]
    E = np.full((n_clients, n_classes), np.nan)
    for c in range(n_classes):
        mask = (y_test == c)
        if mask.sum() == 0:
            continue
        for i in range(n_clients):
            E[i, c] = (preds_test[mask, i] == c).float().mean().item()

    BiasScore = Q - E

    # ---- Evaluation 1: Expertise Correlation ----
    print('\n==> Evaluation 1: Expertise Correlation (Spearman rho_c = corr(q_i(c), E_i(c)) across clients)')
    rho_rows = []
    for c in range(n_classes):
        rho, p = spearmanr(Q[:, c], E[:, c])
        tag = f' <- specialist class' if c in args.specialist_classes else ''
        print(f'  class {c}: rho={rho:.3f} (p={p:.3f}){tag}')
        rho_rows.append({'class': c, 'rho': rho, 'p_value': p, 'is_specialist_class': c in args.specialist_classes})
    rho_df = pd.DataFrame(rho_rows)
    print(f'  mean rho (all classes): {rho_df["rho"].mean():.3f}')
    print(f'  mean rho (specialist classes only): {rho_df[rho_df.is_specialist_class]["rho"].mean():.3f}')

    # ---- Evaluation 2: Specialist Ranking (all classes, specialist classes annotated) ----
    print(f'\n==> Evaluation 2: Ranking (top-{args.top_k} by q_i(c) vs top-{args.top_k} by E_i(c)), all {n_classes} classes')
    rank_rows = []
    for c in range(n_classes):
        gt_top = set(np.argsort(-E[:, c])[:args.top_k].tolist())
        pred_top = set(np.argsort(-Q[:, c])[:args.top_k].tolist())
        overlap = gt_top & pred_top
        precision = len(overlap) / args.top_k
        recall = len(overlap) / min(args.top_k, len(gt_top))
        is_spec_class = c in args.specialist_classes
        true_specialist_id, specialist_rank_by_q = None, None
        extra = ''
        if is_spec_class:
            true_specialist_id = [i for i, (r, sc) in role.items() if r == 'specialist' and sc == c][0]
            specialist_rank_by_q = int(np.argsort(-Q[:, c]).tolist().index(true_specialist_id)) + 1
            extra = f'  (true specialist client {true_specialist_id} ranks #{specialist_rank_by_q} by q_i) <- specialist class'
        print(f'  class {c}: GT top-{args.top_k}={sorted(gt_top)}  q_i top-{args.top_k}={sorted(pred_top)}  '
              f'precision={precision:.2f} recall={recall:.2f}{extra}')
        rank_rows.append({'class': c, 'is_specialist_class': is_spec_class, 'precision': precision, 'recall': recall,
                           'true_specialist_id': true_specialist_id, 'specialist_rank_by_qi': specialist_rank_by_q})
    rank_df = pd.DataFrame(rank_rows)
    print(f'  mean precision (all classes): {rank_df["precision"].mean():.3f}   '
          f'mean precision (specialist classes only): {rank_df[rank_df.is_specialist_class]["precision"].mean():.3f}')

    # ---- Evaluation 3: BiasScore (all classes) ----
    print(f'\n==> Evaluation 3: True Expertise vs Output Bias (BiasScore_i(c) = q_i(c) - E_i(c)), all {n_classes} classes')
    valid = ~np.isnan(BiasScore)
    large_pos = (BiasScore > args.bias_threshold) & valid
    bias_rows = []
    for c in range(n_classes):
        col_valid = valid[:, c]
        col_bias = BiasScore[col_valid, c]
        frac_large = large_pos[:, c].sum() / col_valid.sum() if col_valid.sum() > 0 else np.nan
        tag = ' <- specialist class' if c in args.specialist_classes else ''
        print(f'  class {c}: mean BiasScore={col_bias.mean():.3f}  '
              f'frac(BiasScore>{args.bias_threshold})={frac_large:.3f}{tag}')
        bias_rows.append({'class': c, 'mean_bias_score': col_bias.mean(), 'frac_large_positive_bias': frac_large,
                           'is_specialist_class': c in args.specialist_classes})
    bias_df = pd.DataFrame(bias_rows)
    print(f'  overall mean BiasScore (all client-class pairs): {np.nanmean(BiasScore):.3f}')
    print(f'  overall fraction with BiasScore > {args.bias_threshold}: {large_pos.sum() / valid.sum():.3f}')
    own_class_bias = [BiasScore[i, c] for i, c in zip(specialist_ids, args.specialist_classes)]
    print(f'  specialist BiasScore on their OWN class: {own_class_bias}')

    # ---- Verdict ----
    mean_rho_specialist = rho_df[rho_df.is_specialist_class]["rho"].mean()
    frac_large_bias = large_pos.sum() / valid.sum()
    print('\n==> Verdict:')
    if mean_rho_specialist > 0 and frac_large_bias < 0.3:
        print('  rho_c > 0 (stable) and BiasScore distribution is low -> proceed to Stage 2')
    else:
        print('  weak/no correlation or large BiasScore -> reconsider q_i definition/scoring')

    if args.save_csv:
        rho_df.to_csv(args.save_csv.replace('.csv', '_correlation.csv'), index=False)
        rank_df.to_csv(args.save_csv.replace('.csv', '_ranking.csv'), index=False)
        bias_df.to_csv(args.save_csv.replace('.csv', '_bias.csv'), index=False)
        np.savetxt(args.save_csv.replace('.csv', '_Q_matrix.csv'), Q, delimiter=',')
        np.savetxt(args.save_csv.replace('.csv', '_E_matrix.csv'), E, delimiter=',')
        print(f'\n==> Saved detailed results alongside {args.save_csv}')
