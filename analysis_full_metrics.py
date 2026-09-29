"""
Full metric extraction across logit sets (validation-only; ground-truth labels used only to grade).

Per-specialist-client metrics (Table A): precision, recall(=sensitivity), specificity, balanced accuracy,
false-alarm rate (FPR), AUROC (using the client's own softmax probability as a continuous score),
and vote frequency (q_i, the raw fraction of queries the client answers with that class).

Per-setting aggregator metrics (Table B): rescue set size / rescue ratio (of all queries and of
specialist-class queries), generalist-majority accuracy on specialist vs other classes, Acc_all /
Acc_rescue for CWTM, the original q_i pipeline (as-written and with input tempering 0.3), the
Dawid-Skene diagnostic aggregator, and an ORACLE (label-informed) confusion-matrix ceiling, plus the
fraction of DS's remaining rescue failures explained by "correlated generalist majority outvotes a
correct specialist" (the swamping diagnostic).

Usage: python analysis_full_metrics.py <label>=<logit dir> ...
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from Models.meta_pipeline import MetaPipeline
from Utils.general import get_model

K = 10
SPEC_CLASSES = [3, 4, 9]
NAMES = ['plane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def ds_fit(votes, iters=60, alpha=0.5):
    n, Nc = votes.shape
    T = np.zeros((n, K))
    for i in range(Nc):
        T[np.arange(n), votes[:, i]] += 1
    T /= Nc
    for _ in range(iters):
        prior = T.mean(0) + 1e-6
        prior /= prior.sum()
        conf = np.zeros((Nc, K, K))
        for i in range(Nc):
            for v in range(K):
                conf[i, :, v] = T[votes[:, i] == v].sum(0)
            conf[i] += alpha
            conf[i] /= conf[i].sum(1, keepdims=True)
        logT = np.log(prior)[None, :].repeat(n, 0)
        for i in range(Nc):
            logT += np.log(conf[i][:, votes[:, i]]).T
        logT -= logT.max(1, keepdims=True)
        T = np.exp(logT)
        T /= T.sum(1, keepdims=True)
    prior = T.mean(0) + 1e-6
    prior /= prior.sum()
    return prior, conf


def oracle_fit(votes, labels, alpha=0.5):
    n, Nc = votes.shape
    T = np.eye(K)[labels]
    prior = T.mean(0) + 1e-6
    prior /= prior.sum()
    conf = np.zeros((Nc, K, K))
    for i in range(Nc):
        for v in range(K):
            conf[i, :, v] = T[votes[:, i] == v].sum(0)
        conf[i] += alpha
        conf[i] /= conf[i].sum(1, keepdims=True)
    return prior, conf


def ds_predict_logpost(prior, conf, votes):
    logp = np.log(prior)[None, :].repeat(len(votes), 0)
    for i in range(votes.shape[1]):
        logp += np.log(conf[i][:, votes[:, i]]).T
    return logp


def run(label, path):
    raw = torch.load(f'{path}/logit_testset.pth', map_location='cpu')
    P = F.softmax(torch.stack([x.reshape(-1, K) for x, _ in raw]).float(), -1)
    N = P.shape[1]
    N_GEN = N - len(SPEC_CLASSES)
    SPEC = {N_GEN + si: c for si, c in enumerate(SPEC_CLASSES)}
    yt = torch.tensor([int(t) for _, t in raw])
    y = yt.numpy()
    V = P.argmax(-1).numpy()
    Pn = P.numpy()

    # ---------------- Table A: per-specialist-client classification metrics ----------------
    rowsA = []
    for i, c in SPEC.items():
        vote = V[:, i] == c
        truth = y == c
        tp = (vote & truth).sum(); fp = (vote & ~truth).sum()
        fn = (~vote & truth).sum(); tn = (~vote & ~truth).sum()
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        specificity = tn / (tn + fp)
        bal_acc = (recall + specificity) / 2
        fpr = fp / (fp + tn)
        auroc = roc_auc_score(truth, Pn[:, i, c])
        freq = vote.mean()
        rowsA.append(dict(setting=label, client=i, class_name=NAMES[c],
                          precision=precision, recall_sensitivity=recall, specificity=specificity,
                          balanced_acc=bal_acc, fpr=fpr, auroc=auroc, vote_frequency=freq))

    # ---------------- Table B: aggregator-level metrics ----------------
    pred0 = P.argmax(-1)
    maj = F.one_hot(pred0[:, :N_GEN], K).sum(1).argmax(-1)
    applicable = torch.isin(yt, torch.tensor(SPEC_CLASSES))
    resc = torch.zeros(len(y), dtype=torch.bool)
    for cid, c in SPEC.items():
        resc |= (yt == c) & (maj != c) & (pred0[:, cid] == c)
    macc = np.array([(maj.numpy()[y == c] == c).mean() for c in range(K)])
    sc, oc = SPEC_CLASSES, [c for c in range(K) if c not in SPEC_CLASSES]

    cw = get_model('F_TM2', N, K, 0.25, 224, 1.0, 4, False).to(dev)
    acc = {}
    swamped = []
    for d in range(5):
        rng = np.random.default_rng(1000 + d)
        b = rng.choice(len(y), 500, replace=False)
        e = np.ones(len(y), dtype=bool); e[b] = False
        Ve, ye, re = V[e], y[e], resc.numpy()[e]

        pr_em, cf_em = ds_fit(V[b])
        pr_or, cf_or = oracle_fit(V[b], y[b])
        logp_em = ds_predict_logpost(pr_em, cf_em, Ve)
        logp_or = ds_predict_logpost(pr_or, cf_or, Ve)
        pred_ds = logp_em.argmax(1)
        pred_or = logp_or.argmax(1)

        pcw = torch.cat([cw(P[e][i:i + 250].to(dev), None).argmax(-1).cpu() for i in range(0, int(e.sum()), 250)]).numpy()
        meta = MetaPipeline(K, seed=d, target_maxprob=0.3).set_reference(P[b].to(dev))
        pme = torch.cat([meta(P[e][i:i + 250].to(dev), None).argmax(-1).cpu() for i in range(0, int(e.sum()), 250)]).numpy()
        meta0 = MetaPipeline(K, seed=d, target_maxprob=None).set_reference(P[b].to(dev))
        pm0 = torch.cat([meta0(P[e][i:i + 250].to(dev), None).argmax(-1).cpu() for i in range(0, int(e.sum()), 250)]).numpy()

        for nm, p in (('CWTM', pcw), ('pipeline_as_written', pm0), ('pipeline_tempering0.3', pme),
                     ('dawid_skene', pred_ds), ('oracle_ceiling', pred_or)):
            ok = p == ye
            acc.setdefault(nm, []).append([ok.mean(), ok[re].mean() if re.any() else np.nan])

        fail = re & (pred_ds != ye)
        if fail.sum() > 0:
            n_sw = 0
            for idx in np.where(fail)[0]:
                c_true = ye[idx]; c_wrong = pred_ds[idx]
                si = [cid for cid, cc in SPEC.items() if cc == c_true][0]
                gen_wrong = [i for i in range(N_GEN) if Ve[idx, i] == c_wrong]
                spec_margin = np.log(cf_em[si][c_true, Ve[idx, si]]) - np.log(cf_em[si][c_wrong, Ve[idx, si]])
                bloc_margin = sum(np.log(cf_em[i][c_wrong, Ve[idx, i]]) - np.log(cf_em[i][c_true, Ve[idx, i]]) for i in gen_wrong)
                if bloc_margin > spec_margin:
                    n_sw += 1
            swamped.append(n_sw / fail.sum())

    rowB = dict(setting=label, n_clients=N, n_generalists=N_GEN,
               n_rescue=int(resc.sum()), n_applicable=int(applicable.sum()), n_total=len(y),
               rescue_ratio_of_total=resc.float().mean().item(),
               rescue_ratio_of_applicable=(resc.sum() / applicable.sum()).item(),
               genmaj_acc_specialist_classes=macc[sc].mean(), genmaj_acc_other_classes=macc[oc].mean(),
               swamped_fraction=np.nanmean(swamped) if swamped else np.nan)
    for nm, v in acc.items():
        m = np.nanmean(v, axis=0)
        rowB[f'{nm}_Acc_all'] = m[0]
        rowB[f'{nm}_Acc_rescue'] = m[1]

    return rowsA, rowB


if __name__ == '__main__':
    all_A, all_B = [], []
    for arg in sys.argv[1:]:
        lab, path = arg.split('=', 1)
        print(f'==> {lab}', flush=True)
        rA, rB = run(lab, path)
        all_A += rA
        all_B.append(rB)

    dfA = pd.DataFrame(all_A)
    dfB = pd.DataFrame(all_B)
    pd.set_option('display.width', 250)
    print('\n===== Table A: per-specialist-client classification metrics =====')
    print(dfA.round(3).to_string(index=False))
    print('\n===== Table B: per-setting aggregator metrics =====')
    print(dfB.round(4).to_string(index=False))
    dfA.to_csv('results/C_meta_pipeline/fingerprint_validity/full_metrics_tableA.csv', index=False)
    dfB.to_csv('results/C_meta_pipeline/fingerprint_validity/full_metrics_tableB.csv', index=False)
    print('\nSaved to results/C_meta_pipeline/fingerprint_validity/full_metrics_table{A,B}.csv')
