"""
Evaluate the meta prediction pipeline (Models/meta_pipeline.py) against static aggregators and the
Experiment-A DeepSet_TM2 on the same logits (FL/results/logits/CIFAR10_0.5_M17_90_0.1).

Protocol (labels are used ONLY for evaluation and for the white-box attacks, never by the pipeline):
  - a reference batch B of T0 queries is drawn from the test stream (the pipeline sees only client probabilities on B)
  - every aggregator is evaluated on the remaining test queries (same eval set for all aggregators)
  - the eval queries are split 50/50 into a VALIDATION half and a REPORT half (per draw). Hyper-parameters are chosen on
    'val' only; 'report' is used for the final numbers.
  - clients are permuted with the same order as Experiment A -> adversaries = clients {4,5,7,13} (all generalists)
  - attacks (optional, --attacks) reuse Utils/adversarial.py (sia = paper one-hot, sia_repo = repo version, lma)
  - the whole thing is repeated for `--draws` different reference batches
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from Utils.general import get_model
from Utils.adversarial import sia_attack, loss_maximization_attack
from Models.meta_pipeline import MetaPipeline

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', default='FL/results/logits/CIFAR10_0.5_M17_90_0.1')
parser.add_argument('--a_modelpath', default='results/A_rfi_original/agg_training/cifar10/'
                    'CIFAR10_0.5_M17_90_0.1_DeepSet_M_maxss17_minss13_trim0.25_attackpgd_bbfalse_colludetrue_nsubsets300_cw')
parser.add_argument('--out_dir', default='results/C_meta_pipeline')
parser.add_argument('--tag', default='default')
parser.add_argument('--T0', type=int, default=500)
parser.add_argument('--kappa', type=float, default=5.0)
parser.add_argument('--alpha', type=float, default=0.05, help='global test level (REGIME C)')
parser.add_argument('--alpha_s', type=float, default=0.1, help='BH-FDR target (Step 5)')
parser.add_argument('--n_perm', type=int, default=500)
parser.add_argument('--null', default='pooled', choices=['pooled', 'pernode'])
parser.add_argument('--fallback', default='tm', choices=['mean', 'median', 'tm'],
                    help='what an ABSTAIN returns (tm = CWTM, added to the spec)')
parser.add_argument('--target_maxprob', type=float, default=None,
                    help='input tempering: per-client mean max-prob on B is set to this value (None = off, spec as written)')
parser.add_argument('--draws', type=int, default=5)
parser.add_argument('--n_adv', type=int, default=4)
parser.add_argument('--attacks', nargs='*', default=[], help='e.g. sia sia_repo lma (empty = clean only)')
parser.add_argument('--poison_ref', action='store_true', help='attackers also act on the reference batch B')
parser.add_argument('--no_baselines', action='store_true')
parser.add_argument('--bs', type=int, default=250)
args = parser.parse_args()

SHUFFLE_ORDER = [10, 12, 1, 2, 11, 16, 14, 0, 15, 8, 9, 3, 6, 4, 5, 7, 13]  # from Experiment A test logs (seed 90)
N_GEN, SPEC_CLASSES, K, N = 14, [3, 4, 9], 10, 17
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_probs(datapath):
    raw = torch.load(os.path.join(datapath, 'logit_testset.pth'), map_location='cpu')
    X = torch.stack([x.reshape(-1, K) for x, _ in raw]).float()
    y = torch.tensor([int(t) for _, t in raw])
    return F.softmax(X, dim=-1), y


def rescue_mask(P, y):
    """generalist majority wrong AND the relevant specialist right, on clean outputs (original client order)."""
    pred = P.argmax(-1)
    counts = F.one_hot(pred[:, :N_GEN], K).sum(1)
    maj = counts.argmax(-1)                       # ties -> smallest class (same as np.unique in evaluate_rescue.py)
    m = torch.zeros(len(y), dtype=torch.bool)
    for si, c in enumerate(SPEC_CLASSES):
        m |= (y == c) & (maj != c) & (pred[:, N_GEN + si] == c)
    return m


def batches(X, y):
    out = []
    for i in range(0, len(y), args.bs):
        j = min(i + args.bs, len(y))
        if len(y) - j == 1:                       # avoid a size-1 last batch (static aggregators squeeze())
            j = len(y)
        out.append((X[i:j], torch.ones(j - i, N), y[i:j]))
        if j == len(y):
            break
    return out


@torch.no_grad()
def predict(f, X, y):
    preds, diag = [], []
    for x, mask, _ in batches(X, y):
        out = f(x.to(device), mask.to(device))
        preds.append(out.argmax(-1).cpu())
        if isinstance(f, MetaPipeline):
            diag.append({k: v.cpu() for k, v in f.last.items()})
    d = {k: torch.cat([b[k] for b in diag]) for k in diag[0]} if diag else None
    return torch.cat(preds), d


@torch.no_grad()
def sia_onehot(f, blist, dev, n_adv):
    """SIA-wb exactly as defined in the RFI paper (Table 4): adversaries output a ONE-HOT vector on
    argmax_{k != y} psi(h_1..h_n)."""
    out = []
    for x, mask, y in blist:
        x, mask, y = x.to(dev), mask.to(dev), y.to(dev)
        yp = f(x, mask)
        top2 = torch.topk(yp, 2, dim=-1).indices
        alt = torch.where(top2[:, 0] == y, top2[:, 1], top2[:, 0])
        oh = F.one_hot(alt, K).float()
        xa = x.clone()
        xa[:, -n_adv:] = oh[:, None, :].expand(-1, n_adv, -1)
        out.append((xa.cpu(), mask.cpu(), y.cpu()))
    return out


def attack(f, X, y, kind, n_adv=None):
    n_adv = n_adv or args.n_adv
    fn = {'sia': sia_onehot, 'sia_repo': sia_attack, 'lma': loss_maximization_attack}[kind]
    adv = fn(f, batches(X, y), device, n_adv)
    return torch.cat([a[0] for a in adv])


def group_metrics(pred, yE, rE, spec_y, sel):
    p, y_, r_, s_ = pred[sel], yE[sel], rE[sel], spec_y[sel]
    ok = (p == y_).float()
    return dict(acc_all=ok.mean().item(),
                acc_rescue=ok[r_].mean().item() if r_.any() else float('nan'),
                acc_spec_classes=ok[s_].mean().item(), acc_other_classes=ok[~s_].mean().item(),
                n=int(sel.sum()), n_rescue=int(r_.sum()))


def meta_diag(d, pred, y, base_pred, pos_of, tau):
    reg = d['regime']
    spec_y = torch.zeros(len(y), dtype=torch.bool)
    spec_col = torch.zeros(len(y), dtype=torch.long)
    for si, c in enumerate(SPEC_CLASSES):
        sel = y == c
        spec_y |= sel
        spec_col[sel] = pos_of[N_GEN + si]
    w = d['w']
    ans = reg == 0
    mass_true_spec = w[torch.arange(len(y)), spec_col][spec_y & ans]
    spec_cols = [pos_of[N_GEN + si] for si in range(3)]
    return {
        'frac_B_answered': ans.float().mean().item(),
        'frac_C_global_abstain': (reg == 1).float().mean().item(),
        'frac_D_selection_abstain': (reg == 2).float().mean().item(),
        'acc_on_answered': (pred[ans] == y[ans]).float().mean().item() if ans.any() else float('nan'),
        'cstar_eq_label_all': (d['cstar'] == y).float().mean().item(),
        'cstar_eq_label_answered': (d['cstar'][ans] == y[ans]).float().mean().item() if ans.any() else float('nan'),
        'mass_on_true_specialist': mass_true_spec.mean().item() if len(mass_true_spec) else float('nan'),
        'mass_on_any_specialist': w[:, spec_cols].sum(1)[ans].mean().item() if ans.any() else float('nan'),
        'gain_vs_cwtm': int(((base_pred != y) & (pred == y)).sum()),
        'harm_vs_cwtm': int(((base_pred == y) & (pred != y)).sum()),
        'median_abs_sigma_max': d['sigma'].abs().max(1).values.median().item(),
        'median_pglobal': d['p_global'].median().item(),
        'mean_tau': float(tau.mean()) if tau is not None else 1.0,
        'max_tau': float(tau.max()) if tau is not None else 1.0,
    }


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    P, y = load_probs(args.datapath)
    resc_all = rescue_mask(P, y)
    order = SHUFFLE_ORDER
    pos_of = {cid: j for j, cid in enumerate(order)}
    Pshuf = P[:, order]
    print(f'==> {len(y)} test queries, rescue set {int(resc_all.sum())}; mean client max-prob {P.max(-1).values.mean():.4f}; '
          f'target_maxprob={args.target_maxprob} fallback={args.fallback} kappa={args.kappa} n_perm={args.n_perm} '
          f'alpha={args.alpha} alpha_s={args.alpha_s} T0={args.T0}')

    aggs = {}
    if not args.no_baselines:
        for nm in ['F_Avg', 'F_Median2', 'F_TM2', 'F_Geo_Median']:
            aggs[nm] = get_model(nm, N, K, 0.25, 224, 1.0, args.n_adv, False).to(device)
        ds = get_model('DeepSet_TM2', N, K, 0.25, 224, 1.0, args.n_adv, False)
        ds.load_state_dict(torch.load(os.path.join(args.a_modelpath, 'model.pth'), map_location='cpu'))
        aggs['DeepSet_TM2(A)'] = ds.to(device).eval()
    cw = get_model('F_TM2', N, K, 0.25, 224, 1.0, args.n_adv, False).to(device)

    rows, drows = [], []
    for draw in range(args.draws):
        rng = np.random.default_rng(1000 + draw)
        b_idx = torch.tensor(rng.choice(len(y), size=args.T0, replace=False))
        e_mask = torch.ones(len(y), dtype=torch.bool)
        e_mask[b_idx] = False
        XB, yB = Pshuf[b_idx], y[b_idx]
        XE, yE, rE = Pshuf[e_mask], y[e_mask], resc_all[e_mask]
        spec_y = torch.isin(yE, torch.tensor(SPEC_CLASSES))
        val_sel = torch.zeros(len(yE), dtype=torch.bool)
        val_sel[torch.tensor(np.random.default_rng(2000 + draw).permutation(len(yE))[:len(yE) // 2])] = True
        splits = {'val': val_sel, 'report': ~val_sel}

        def make_meta(Xref):
            return MetaPipeline(K, kappa=args.kappa, alpha=args.alpha, alpha_s=args.alpha_s, n_perm=args.n_perm,
                                null=args.null, seed=draw, fallback=args.fallback,
                                target_maxprob=args.target_maxprob).set_reference(Xref.to(device))

        meta = make_meta(XB)
        allaggs = dict(aggs)
        allaggs['Meta'] = meta

        for name, f in allaggs.items():
            conds = {'clean': XE}
            for kind in args.attacks:
                conds[kind] = attack(f, XE, yE, kind).cpu()
            for cond, Xc in conds.items():
                pred, d = predict(f, Xc, yE)
                for sp, sel in splits.items():
                    rows.append(dict(draw=draw, agg=name, cond=cond, split=sp, **group_metrics(pred, yE, rE, spec_y, sel)))
                if d is not None:
                    base_pred, _ = predict(cw, Xc, yE)
                    drows.append(dict(draw=draw, agg=name, cond=cond,
                                      **meta_diag(d, pred, yE, base_pred, pos_of, meta.tau)))

        if args.poison_ref:                      # attackers also act on the reference batch (crafted against CWTM)
            for kind in args.attacks:
                metap = make_meta(attack(cw, XB, yB, kind))
                Xa = attack(metap, XE, yE, kind).cpu()
                pred, d = predict(metap, Xa, yE)
                for sp, sel in splits.items():
                    rows.append(dict(draw=draw, agg='Meta(poisoned_ref)', cond=kind, split=sp,
                                     **group_metrics(pred, yE, rE, spec_y, sel)))
                base_pred, _ = predict(cw, Xa, yE)
                drows.append(dict(draw=draw, agg='Meta(poisoned_ref)', cond=kind,
                                  **meta_diag(d, pred, yE, base_pred, pos_of, metap.tau)))
        print(f'   draw {draw} done', flush=True)

    df = pd.DataFrame(rows)
    dd = pd.DataFrame(drows)
    df.to_csv(os.path.join(args.out_dir, f'{args.tag}_metrics.csv'), index=False)
    dd.to_csv(os.path.join(args.out_dir, f'{args.tag}_diag.csv'), index=False)

    pd.set_option('display.width', 220)
    cols = ['acc_all', 'acc_rescue', 'acc_spec_classes', 'acc_other_classes']
    g = df.groupby(['split', 'agg', 'cond'])[cols].mean().round(4)
    print('\n==> mean over reference-batch draws')
    print(g.to_string())
    if len(dd):
        print('\n==> Meta diagnostics (mean over draws)')
        print(dd.groupby(['agg', 'cond']).mean(numeric_only=True).drop(columns='draw').round(4).T.to_string())


if __name__ == '__main__':
    main()
