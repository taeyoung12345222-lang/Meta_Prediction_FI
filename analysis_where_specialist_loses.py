"""
For rescue-set queries specifically, decompose WHERE the original q_i (+ tempering) pipeline fails:
  A) REGIME C (global abstain): the permutation test on s_t(c*) itself failed -- did c* even equal
     the true class before being rejected?
  B) REGIME D (selection abstain): c* passed the global test, but no client's weight survived BH-FDR
     (everyone, including the specialist, got filtered out) -- "the expert wanted to speak but was
     silenced by the statistical test."
  C) c* != true class (wrong candidate entirely, regardless of regime): the specialist's opinion about
     its own class was never even considered, because some OTHER class won Step 4.
  D) c* == true class AND answered (REGIME B), but the final weighted average still lands on the
     wrong class: the specialist got to speak but was outvoted in the weighted combination
     ("majority swamping" at the Step 6 combination level).
  E) c* == true class, answered, and correct: success.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

from Models.meta_pipeline import MetaPipeline

K = 10
SPEC_CLASSES = [3, 4, 9]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def run(label, path, target=0.3):
    raw = torch.load(f'{path}/logit_testset.pth', map_location='cpu')
    P = F.softmax(torch.stack([x.reshape(-1, K) for x, _ in raw]).float(), -1)
    N = P.shape[1]
    N_GEN = N - len(SPEC_CLASSES)
    SPEC = {N_GEN + si: c for si, c in enumerate(SPEC_CLASSES)}
    yt = torch.tensor([int(t) for _, t in raw])
    y = yt.numpy()
    pred0 = P.argmax(-1)
    maj = F.one_hot(pred0[:, :N_GEN], K).sum(1).argmax(-1)
    resc = torch.zeros(len(y), dtype=torch.bool)
    for cid, c in SPEC.items():
        resc |= (yt == c) & (maj != c) & (pred0[:, cid] == c)

    cats = dict(regimeC_cstar_would_be_right=0, regimeC_cstar_wrong=0,
               regimeD_cstar_right_but_silenced=0, regimeD_cstar_wrong=0,
               wrong_candidate_answered=0, right_candidate_but_outvoted=0, success=0)
    n_total = 0
    spec_weight_when_right_candidate = []

    for d in range(5):
        rng = np.random.default_rng(1000 + d)
        b = rng.choice(len(y), 500, replace=False)
        e = np.ones(len(y), dtype=bool)
        e[b] = False
        e_resc = e & resc.numpy()
        idxs = np.where(e_resc)[0]
        if len(idxs) == 0:
            continue
        meta = MetaPipeline(K, seed=d, target_maxprob=target).set_reference(P[b].to(dev))
        for i0 in range(0, len(idxs), 250):
            chunk = idxs[i0:i0 + 250]
            x = P[chunk].to(dev)
            out = meta(x, None)
            pred = out.argmax(-1).cpu().numpy()
            reg = meta.last['regime'].cpu().numpy()
            cst = meta.last['cstar'].cpu().numpy()
            w = meta.last['w'].cpu().numpy()
            yy = y[chunk]
            for k in range(len(chunk)):
                n_total += 1
                right_cand = cst[k] == yy[k]
                if reg[k] == 1:            # global abstain
                    cats['regimeC_cstar_would_be_right' if right_cand else 'regimeC_cstar_wrong'] += 1
                elif reg[k] == 2:          # selection abstain (BH-FDR filtered everyone out)
                    cats['regimeD_cstar_right_but_silenced' if right_cand else 'regimeD_cstar_wrong'] += 1
                else:                       # answered
                    if not right_cand:
                        cats['wrong_candidate_answered'] += 1
                    elif pred[k] == yy[k]:
                        cats['success'] += 1
                    else:
                        cats['right_candidate_but_outvoted'] += 1
                        # how much weight did the specialist for this class actually get?
                        si = [cid for cid, cc in SPEC.items() if cc == yy[k]][0]
                        spec_weight_when_right_candidate.append(w[k, si])

    print(f'\n== {label} (target_maxprob={target}, n_rescue_eval={n_total}) ==')
    for k, v in cats.items():
        print(f'  {k:38s} {v:5d}  ({v / n_total:.1%})')
    if spec_weight_when_right_candidate:
        arr = np.array(spec_weight_when_right_candidate)
        print(f'  when candidate==true class but final answer wrong: specialist\'s own weight share '
              f'mean={arr.mean():.3f} median={np.median(arr):.3f} (1/{N}={1/N:.3f} would be "equal share")')


for arg in sys.argv[1:]:
    lab, path = arg.split('=', 1)
    run(lab, path)
