"""
Meta prediction pipeline (meta_prediction_pipeline 8/24), Steps 0-5 + Step 6 (weighted average).

Data-free / zero-knowledge: only the clients' output probabilities on a reference batch B and
on the current query are used. No labels, no client self-report.

Step 0  q_i(c) = S_i(c)/n over B, smoothed qbar_i = (S_i+1)/(n+K), r_i = clip(qbar_g/qbar_i, 1/kappa, kappa)
Step 2  m_i(c|x) = p_i(c|x) r_i(c) / Z_i,  Z_i = sum_c' p_i(c'|x) r_i(c')
Step 3  s_t(c) = P_t(c) - M_t(c)
Step 4  c* = argmax s_t ; global permutation test (q_i <-> p_i mapping shuffled) -> p_global; > alpha => REGIME C
Step 5  sigma_i = p_i(c*) - m_i(c*), per-node permutation p-values, BH-FDR -> rho_i,
        w_i ~ max(sigma_i,0) * max(0, 1 - rho_i/alpha_S); all zero => REGIME D
Step 6  final = sum_i w_i p_i   (ASSUMPTION: Step 6 is not in the source document; uses the raw client probabilities)

Additions that are NOT in the source document (decided with the author, 2026-09-21):
  * fallback: an ABSTAIN (REGIME C/D) returns the coordinate-wise trimmed mean (trim 25%) of the raw client
    probabilities ('tm', default). 'mean' / 'median' are kept only for ablations.
  * input tempering (option target_maxprob): before Steps 0-5 every client's probabilities are softened,
    p_i <- normalize(p_i^(1/tau_i)), with tau_i chosen on the reference batch B (label-free) so that client i's mean
    max-probability on B equals `target_maxprob` (clients already softer than the target keep tau_i = 1).
    This is a stop-gap for the saturation of Step 2 for over-confident clients (Z_i cancels the prior correction).

`last['regime']` codes: 0 = B (answered), 1 = C (global abstain), 2 = D (selection abstain).
"""
import torch
import torch.nn as nn


class MetaPipeline(nn.Module):
    def __init__(self, n_classes, kappa=5.0, alpha=0.05, alpha_s=0.1, n_perm=500,
                 null='pooled', seed=0, fallback='tm', target_maxprob=None):
        super().__init__()
        assert null in ('pooled', 'pernode')
        assert fallback in ('mean', 'median', 'tm')
        self.K = n_classes
        self.kappa = kappa
        self.alpha = alpha
        self.alpha_s = alpha_s
        self.n_perm = n_perm
        self.null = null
        self.fallback = fallback
        self.target = target_maxprob
        self.gen = torch.Generator().manual_seed(seed)
        self.r = None
        self.tau = None
        self.last = {}

    # ------------------------------------------------------------------ tempering (label-free)
    def _temper(self, x):
        if self.tau is None:
            return x
        lp = torch.log(x.clamp(min=1e-12))
        return torch.softmax(lp / self.tau.to(x.device)[None, :, None], dim=-1)

    @torch.no_grad()
    def _fit_tau(self, P_B):
        """per-client tau_i with mean max-prob on B equal to the target (bisection on log tau)."""
        n, N, K = P_B.shape
        lp = torch.log(P_B.clamp(min=1e-12))
        lo = torch.zeros(N, device=P_B.device)
        hi = torch.full((N,), float(torch.log(torch.tensor(1000.0))), device=P_B.device)
        for _ in range(40):
            mid = (lo + hi) / 2
            mp = torch.softmax(lp / torch.exp(mid)[None, :, None], dim=-1).max(-1).values.mean(0)
            too_sharp = mp > self.target
            lo = torch.where(too_sharp, mid, lo)
            hi = torch.where(too_sharp, hi, mid)
        tau = torch.exp(hi)
        mp1 = torch.softmax(lp, dim=-1).max(-1).values.mean(0)
        return torch.where(mp1 <= self.target, torch.ones_like(tau), tau)

    # ------------------------------------------------------------------ Step 0
    @torch.no_grad()
    def set_reference(self, P_B):
        """P_B: [T0, N, K] raw client probabilities on the reference batch."""
        self.tau = self._fit_tau(P_B) if self.target is not None else None
        P_B = self._temper(P_B)
        n, N, K = P_B.shape
        S = P_B.sum(0)                                  # [N, K]
        qbar = (S + 1.0) / (n + K)                      # smoothed prior
        qbar_g = qbar.mean(0)                           # mean of smoothed == smoothed mean (same eps)
        self.r = (qbar_g[None, :] / qbar).clamp(1.0 / self.kappa, self.kappa)   # [N, K]
        self.n_ref = n
        return self

    # ------------------------------------------------------------------ Steps 1-6
    @torch.no_grad()
    def forward(self, x_raw, mask=None):
        B, N, K = x_raw.shape
        dev = x_raw.device
        x = self._temper(x_raw)                                         # inputs of Steps 2-5
        r = self.r.to(dev)
        P = x.mean(1)                                                   # [B, K]
        Z = (x * r[None]).sum(-1, keepdim=True)                         # [B, N, 1]
        m = x * r[None] / Z                                             # [B, N, K]
        s = P - m.mean(1)                                               # [B, K]
        s_obs, cstar = s.max(-1)                                        # [B]

        idx = cstar[:, None, None].expand(B, N, 1)
        xc = x.gather(2, idx).squeeze(2)                                # [B, N]
        sigma = xc - m.gather(2, idx).squeeze(2)                        # [B, N]

        # permutation nulls: client i is paired with the prior of client perm(i)
        perms = torch.stack([torch.randperm(N, generator=self.gen) for _ in range(self.n_perm)]).to(dev)
        r_pi = r[perms]                                                 # [Pm, N, K]
        Zp = torch.einsum('bnk,pnk->bpn', x, r_pi)                      # [B, Pm, N]
        m_pi = x[:, None] * r_pi[None] / Zp[..., None]                  # [B, Pm, N, K]
        s_null = (P[:, None, :] - m_pi.mean(2)).max(-1).values          # [B, Pm]  (max over c: selection-aware)
        p_global = (1.0 + (s_null >= s_obs[:, None]).sum(1).float()) / (1.0 + self.n_perm)

        m_pi_c = m_pi.gather(3, cstar[:, None, None, None].expand(B, self.n_perm, N, 1)).squeeze(3)  # [B, Pm, N]
        sigma_null = xc[:, None, :] - m_pi_c                            # [B, Pm, N]
        if self.null == 'pooled':
            nul = sigma_null.reshape(B, -1)                             # [B, Pm*N]
            pval = (1.0 + (nul[:, None, :] >= sigma[:, :, None]).sum(-1).float()) / (1.0 + nul.shape[1])
        else:
            nul = sigma_null.permute(0, 2, 1)                           # [B, N, Pm]
            pval = (1.0 + (nul >= sigma[:, :, None]).sum(-1).float()) / (1.0 + self.n_perm)

        # BH-FDR adjusted p-values rho_i = min_{k'>=rank(i)} N/k' * p_(k')
        sp, order = pval.sort(dim=1)
        k = torch.arange(1, N + 1, device=dev).float()
        adj = sp * N / k[None]
        adj = torch.flip(torch.cummin(torch.flip(adj, [1]), dim=1).values, [1])
        rho = torch.empty_like(pval).scatter_(1, order, adj)

        w_raw = sigma.clamp(min=0) * (1.0 - rho / self.alpha_s).clamp(min=0)
        tot = w_raw.sum(1)
        w = w_raw / tot.clamp(min=1e-12)[:, None]

        regime = torch.zeros(B, dtype=torch.long, device=dev)
        regime[tot <= 0] = 2
        regime[p_global > self.alpha] = 1                               # C takes precedence (checked first in Step 4)
        out_B = (w[..., None] * x_raw).sum(1)                           # Step 6 on the raw client probabilities
        if self.fallback == 'mean':
            fb = x_raw.mean(1)
        elif self.fallback == 'median':
            fb = x_raw.median(dim=1).values
        else:                                                           # coordinate-wise trimmed mean, trim 25%
            k_trim = int(N * 0.25)
            fb = x_raw.sort(dim=1).values[:, k_trim:N - k_trim].mean(1)
        out = torch.where((regime == 0)[:, None], out_B, fb)

        self.last = dict(regime=regime, cstar=cstar, w=w, p_global=p_global, sigma=sigma,
                         rho=rho, pval=pval, s_obs=s_obs, tot=tot)
        return out
