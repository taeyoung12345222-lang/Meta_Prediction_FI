"""
Quick supplementary check: does the tempered pipeline's rescue gain come at the cost of accuracy on
NON-specialist classes (as it did in the earlier ViT experiments, e.g. plane->ship, bird->deer)?
Also reports per-draw variance of Acc_rescue to check how stable the headline number is.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

from Models.meta_pipeline import MetaPipeline
from Utils.general import get_model

K = 10
SPEC_CLASSES = [3, 4, 9]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def run(label, path):
    raw = torch.load(f'{path}/logit_testset.pth', map_location='cpu')
    P = F.softmax(torch.stack([x.reshape(-1, K) for x, _ in raw]).float(), -1)
    N = P.shape[1]
    yt = torch.tensor([int(t) for _, t in raw])
    spec_y = torch.isin(yt, torch.tensor(SPEC_CLASSES))
    cw = get_model('F_TM2', N, K, 0.25, 224, 1.0, 4, False).to(dev)

    rows = {'CWTM': [], 'tempering0.3': []}
    for d in range(5):
        rng = np.random.default_rng(1000 + d)
        b = rng.choice(len(yt), 500, replace=False)
        e = np.ones(len(yt), dtype=bool); e[b] = False
        e_t = torch.from_numpy(e)
        pcw = torch.cat([cw(P[e_t][i:i + 250].to(dev), None).argmax(-1).cpu() for i in range(0, int(e.sum()), 250)])
        meta = MetaPipeline(K, seed=d, target_maxprob=0.3).set_reference(P[b].to(dev))
        pme = torch.cat([meta(P[e_t][i:i + 250].to(dev), None).argmax(-1).cpu() for i in range(0, int(e.sum()), 250)])
        ye, se = yt[e_t], spec_y[e_t]
        rescue_e = None  # computed below globally
        for nm, p in (('CWTM', pcw), ('tempering0.3', pme)):
            ok = (p == ye).float()
            rows[nm].append([ok[se].mean().item(), ok[~se].mean().item(), ok.mean().item()])

    print(f'\n== {label} ==')
    for nm in rows:
        arr = np.array(rows[nm])
        print(f'  {nm:14s} specialist-class acc {arr[:,0].mean():.4f} (std {arr[:,0].std():.3f}) | '
              f'other-class acc {arr[:,1].mean():.4f} (std {arr[:,1].std():.3f}) | overall {arr[:,2].mean():.4f}')
    delta_other = np.array(rows['tempering0.3'])[:, 1].mean() - np.array(rows['CWTM'])[:, 1].mean()
    print(f'  -> tempering0.3 vs CWTM on OTHER classes: {delta_other:+.4f}')


for arg in sys.argv[1:]:
    lab, path = arg.split('=', 1)
    run(lab, path)
