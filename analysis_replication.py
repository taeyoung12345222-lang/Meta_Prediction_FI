"""
Replication of the fingerprint diagnostics on other logit sets (validation-only; the pipeline code is not modified).
Usage: python analysis_replication.py <label>=<logit dir> ...
For every set it reports
  (1) specialist (client, class) pairs: true precision / recall / false-alarm
  (2) look-alikes: generalist pairs that are as 'high recall / not more precise' as the specialists
  (3) group blind spot: generalist-majority accuracy on the specialist classes vs the other classes
  (4) how the label-free statistics relate to true precision (frequency vs Dawid-Skene)
  (5) aggregator comparison, T0=500, 5 reference-batch draws: CWTM / current pipeline (target 0.3, unchanged) / Dawid-Skene posterior
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from Models.meta_pipeline import MetaPipeline
from Utils.general import get_model

K = 10
SPEC_CLASSES = [3, 4, 9]   # cat, deer, truck -- always the last len(SPEC_CLASSES) client ids, in this order
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


def ds_predict(prior, conf, votes):
    logp = np.log(prior)[None, :].repeat(len(votes), 0)
    for i in range(votes.shape[1]):
        logp += np.log(conf[i][:, votes[:, i]]).T
    return logp.argmax(1)


def run(label, path):
    raw = torch.load(f'{path}/logit_testset.pth', map_location='cpu')
    P = F.softmax(torch.stack([x.reshape(-1, K) for x, _ in raw]).float(), -1)
    N = P.shape[1]
    N_GEN = N - len(SPEC_CLASSES)
    SPEC = {N_GEN + si: c for si, c in enumerate(SPEC_CLASSES)}
    yt = torch.tensor([int(t) for _, t in raw])
    y = yt.numpy()
    V = P.argmax(-1).numpy()
    print(f'\n################ {label}  ({path})  {N} clients ({N_GEN} generalists + {len(SPEC_CLASSES)} specialists), '
          f'mean client max-prob {P.max(-1).values.mean():.3f}')
    cacc = (torch.tensor(V) == yt[:, None]).float().mean(0).numpy()
    print(f'(0) client quality: mean generalist acc {cacc[:N_GEN].mean():.3f} (min {cacc[:N_GEN].min():.3f}, max {cacc[:N_GEN].max():.3f}) | '
          f'specialists {np.round(cacc[N_GEN:], 3).tolist()} | mean-probability ensemble acc {(P.mean(1).argmax(-1) == yt).float().mean():.4f}')

    prec = np.full((N, K), np.nan)
    rec = np.zeros((N, K))
    fa = np.zeros((N, K))
    nv = np.zeros((N, K), dtype=int)
    for i in range(N):
        for c in range(K):
            v = V[:, i] == c
            nv[i, c] = v.sum()
            if v.sum() >= 20:
                prec[i, c] = (y[v] == c).mean()
            rec[i, c] = (V[y == c, i] == c).mean()
            fa[i, c] = (V[y != c, i] == c).mean()

    print('(1) specialist pairs: client/class | precision | recall | false-alarm')
    for i, c in SPEC.items():
        print(f'    {i}/{NAMES[c]:5s} | {prec[i, c]:.3f} | {rec[i, c]:.3f} | {fa[i, c]:.3f}')
    sp_prec = np.array([prec[i, c] for i, c in SPEC.items()])
    look = [(i, c) for i in range(N_GEN) for c in range(K)
            if rec[i, c] >= 0.9 and not np.isnan(prec[i, c]) and prec[i, c] <= sp_prec.max() + 0.05 and nv[i, c] >= 500]
    print(f'(2) generalist look-alikes (recall >= 0.9, precision <= {sp_prec.max() + 0.05:.2f}, >= 500 votes): {len(look)} pairs')
    for i, c in sorted(look, key=lambda t: -nv[t])[:5]:
        print(f'    {i}/{NAMES[c]:5s} | precision {prec[i, c]:.3f} | recall {rec[i, c]:.3f} | false-alarm {fa[i, c]:.3f} | votes {nv[i, c]}')

    maj = np.array([np.bincount(V[j, :N_GEN], minlength=K).argmax() for j in range(len(y))])
    macc = np.array([(maj[y == c] == c).mean() for c in range(K)])
    sc = list(SPEC.values())
    oc = [c for c in range(K) if c not in sc]
    print(f'(3) generalist-majority accuracy: specialist classes {np.round(macc[sc], 3).tolist()} | other classes mean {macc[oc].mean():.3f} (min {macc[oc].min():.3f})')

    sp = []
    for T0 in (500, 2000):
        rho_f, rho_d = [], []
        for d in range(5):
            b = np.random.default_rng(1000 + d).choice(len(y), T0, replace=False)
            PB = P[b].numpy()
            q = PB.mean(0)
            freq = (q + 1e-3) / (q.mean(0, keepdims=True) + 1e-3)
            dsp = ds_fit(V[b])
            prior, conf = dsp
            dprec = np.array([[prior[c] * conf[i, c, c] / (prior * conf[i, :, c]).sum() for c in range(K)] for i in range(N)])
            ok = ~np.isnan(prec) & (nv >= 20)
            rho_f.append(spearmanr(freq[ok], prec[ok])[0])
            rho_d.append(spearmanr(dprec[ok], prec[ok])[0])
        sp.append((T0, np.mean(rho_f), np.mean(rho_d)))
    print('(4) Spearman with true precision (all pairs): ' + '; '.join(f'T0={t}: frequency {a:+.2f}, Dawid-Skene {b:+.2f}' for t, a, b in sp))

    resc_pred = P.argmax(-1)
    majr = F.one_hot(resc_pred[:, :N_GEN], K).sum(1).argmax(-1)
    resc = torch.zeros(len(y), dtype=torch.bool)
    for cid, c in SPEC.items():
        resc |= (yt == c) & (majr != c) & (resc_pred[:, cid] == c)
    spec_y = torch.isin(yt, torch.tensor(sc))
    cw = get_model('F_TM2', N, K, 0.25, 224, 1.0, 4, False).to(dev)
    acc = {}
    for d in range(5):
        b = np.random.default_rng(1000 + d).choice(len(y), 500, replace=False)
        e = torch.ones(len(y), dtype=torch.bool)
        e[b] = False
        pr, cf = ds_fit(V[b])
        pds = torch.tensor(ds_predict(pr, cf, V[e.numpy()]))
        pcw = torch.cat([cw(P[e][i:i + 250].to(dev), None).argmax(-1).cpu() for i in range(0, int(e.sum()), 250)])
        meta = MetaPipeline(K, seed=d, target_maxprob=0.3).set_reference(P[b].to(dev))
        pme = torch.cat([meta(P[e][i:i + 250].to(dev), None).argmax(-1).cpu() for i in range(0, int(e.sum()), 250)])
        meta0 = MetaPipeline(K, seed=d, target_maxprob=None).set_reference(P[b].to(dev))
        pm0 = torch.cat([meta0(P[e][i:i + 250].to(dev), None).argmax(-1).cpu() for i in range(0, int(e.sum()), 250)])
        for nm, p in (('CWTM', pcw), ('pipeline as written', pm0), ('pipeline + tempering 0.3', pme), ('Dawid-Skene', pds)):
            ok = (p == yt[e]).float()
            acc.setdefault(nm, []).append([ok.mean().item(), ok[resc[e]].mean().item() if resc[e].any() else float('nan'),
                                           ok[spec_y[e]].mean().item(), ok[~spec_y[e]].mean().item()])
    print(f'(5) aggregators, T0=500 (rescue set size {int(resc.sum())}): Acc_all | Acc_rescue | specialist classes | other classes')
    for nm, v in acc.items():
        m = np.nanmean(v, axis=0)
        print(f'    {nm:25s} | {m[0]:.4f} | {m[1]:.4f} | {m[2]:.4f} | {m[3]:.4f}')


for arg in sys.argv[1:]:
    lab, path = arg.split('=', 1)
    run(lab, path)
