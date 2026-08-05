# Round 6.4 Protocol: Dual Clean-label Audit

## Motivation

Round 6.3 found

$$
\operatorname{ASR}(M_p,T_{\mathrm{shuffle}})
\geq
\operatorname{ASR}(M_p,T)
$$

for every pilot. This falsified the claim that the current generator had learned victim-specific contextual binding, but it did not rule out a generic transferable clean-label trigger.

Round 6.4 treats these as distinct hypotheses.

## Branch A: generic trigger

A generic trigger should remain effective after victim shuffling. Generator optimization adds

$$
\mathcal L_{\mathrm{generic}}
=
\lambda_s\operatorname{CE}\left(f_p(v,T_{\pi(v)}),y_t\right)
+
\lambda_c\left\|T_v-\overline T\right\|_F^2
$$

The formal gate is

- clean no-trigger ASR $\leq0.10$
- maximum clean-model triggered ASR $\leq0.20$
- poisoned-model no-trigger ASR $\leq0.20$
- minimum of matched and shuffled poisoned ASR $\geq0.80$
- generic DiD $\geq0.40$

## Branch B: contextual trigger

A contextual trigger should depend on the victim. Let $T_v$ be the trigger generated for victim $v$ and $T_{\pi(v)}$ a trigger generated for another victim. The added pair loss is

$$
\mathcal L_{\mathrm{pair}}
=
\frac1{|C|}
\sum_{v\in C}
\max\left\{
0,
 m-s_t(v,T_v)+s_t(v,T_{\pi(v)})
\right\}
$$

where $C$ is the held-out calibration set and $s_t$ is the poisoned surrogate target logit.

To discourage prototype collapse, the generator also preserves pairwise context geometry:

$$
\mathcal L_{\mathrm{relation}}
=
\left\|
\operatorname{cos}(\overline T_i,\overline T_j)
-
\operatorname{cos}(x_i,x_j)
\right\|_2^2
$$

and enforces a small minimum feature variance.

The contextual gate is

- clean no-trigger ASR $\leq0.10$
- clean-model matched-trigger ASR $\leq0.20$
- poisoned-model no-trigger ASR $\leq0.20$
- poisoned-model shuffled-trigger ASR $\leq0.20$
- poisoned-model matched-trigger ASR $\geq0.80$
- contextual binding gap $\geq0.40$

## Statistical reporting

For each valid branch report:

- Full ASR and all controls
- Clean accuracy
- admission gap
- spectral AUROC
- spectral AUPRC
- FPR@95TPR
- permutation $p$-value

Invalid attacks remain diagnostic runs and are excluded from defense claims.
