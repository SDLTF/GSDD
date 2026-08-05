# GSDD-v0.1 Theory Specification

## 1. Research object

Let the possibly poisoned graph be

$$
\widetilde G=(V,\widetilde E,\widetilde X)
$$

Let $V_p\subseteq V_l$ denote the unknown poisoned labeled nodes. A supervised encoder $f_s$ is trained with possibly manipulated labels, while a self-supervised encoder $f_u$ is trained without those labels.

The central hypothesis is not that every trigger is high-frequency. It is:

$$
D_{\mathrm{transfer}}(v)
\text{ is stochastically larger for }v\in V_p
$$

where $D_{\mathrm{transfer}}$ measures abnormal cross-model differences in multi-band graph-signal amplification.

## 2. Laplacian and Bernstein filter bank

Use the symmetric normalized Laplacian

$$
L=I-D^{-1/2}AD^{-1/2}
$$

whose eigenvalues lie in $[0,2]$. Define $t=\lambda/2\in[0,1]$. For $B$ bands, let $n=B-1$ and define

$$
g_b(\lambda)
=
\binom{n}{b}t^b(1-t)^{n-b}
$$

for $b=0,1,\ldots,n$.

The filters satisfy the partition of unity

$$
\sum_{b=0}^{n}g_b(\lambda)=1
$$

They are polynomial filters, so no eigendecomposition is required.

For the default $B=4$:

$$
g_0(L)=\left(I-\frac L2\right)^3
$$

$$
g_1(L)=3\frac L2\left(I-\frac L2\right)^2
$$

$$
g_2(L)=3\left(\frac L2\right)^2\left(I-\frac L2\right)
$$

$$
g_3(L)=\left(\frac L2\right)^3
$$

## 3. Node-level band energy

For a graph signal $Z\in\mathbb R^{N\times q}$, define

$$
R_b(v;Z)
=
\left\|e_v^\top g_b(L)Z\right\|_2^2
$$

and

$$
p_b(v;Z)
=
\frac{R_b(v;Z)+\varepsilon}
{\sum_{c=0}^{B-1}R_c(v;Z)+B\varepsilon}
$$

Thus

$$
p(v;Z)\in\Delta^{B-1}
$$

is a node-specific graph-frequency distribution.

## 4. H3: model spectral-distribution discrepancy

Let $H_s^{(\ell)}$ and $H_u^{(\ell)}$ be hidden representations from the supervised and self-supervised encoders.

Define

$$
D_{\mathrm{JS}}^{(\ell)}(v)
=
D_{\mathrm{JS}}
\left(
p(v;H_s^{(\ell)})
\parallel
p(v;H_u^{(\ell)})
\right)
$$

The raw pointwise H3 statistic is the vector

$$
s_{\mathrm{model}}(v)
=
\left(
D_{\mathrm{JS}}^{(1)}(v),
\ldots,
D_{\mathrm{JS}}^{(L)}(v)
\right)
$$

A second H3 statistic mirrors DShield's same-label relation discrepancy. For nodes $i,j$ with the same observed label, let

$$
d_u^{(\ell)}(i,j)
=
\left\|p(i;H_u^{(\ell)})-p(j;H_u^{(\ell)})\right\|_2
$$

$$
d_s^{(\ell)}(i,j)
=
\left\|p(i;H_s^{(\ell)})-p(j;H_s^{(\ell)})\right\|_2
$$

Then

$$
r^{(\ell)}(i)
=
\frac{1}{|C_i|-1}
\sum_{j\in C_i, j\ne i}
\max\left\{d_u^{(\ell)}(i,j)-d_s^{(\ell)}(i,j),0\right\}
$$

This detects nodes that are spectrally far from their observed class in the label-free view but are pulled toward that class by supervised learning. Robust calibration produces `score_model_js` and `score_spectral_relation`.

## 5. H4: spectral-transfer discrepancy

A hidden representation can have a different overall norm merely because the two training objectives use different scales. The code therefore computes band energy per channel:

$$
\overline R_b(v;H)
=
\frac{R_b(v;H)}{\dim(H)}
$$

For model $m\in\{s,u\}$, define the log gain

$$
T_{m,b}^{(\ell)}(v)
=
\log
\frac{
\overline R_b(v;H_m^{(\ell)})+\varepsilon
}{
\overline R_b(v;X)+\varepsilon
}
$$

and cross-model gain difference

$$
\Delta T_b^{(\ell)}(v)
=
T_{s,b}^{(\ell)}(v)
-
T_{u,b}^{(\ell)}(v)
$$

The raw H4 feature is

$$
s_{\mathrm{transfer}}(v)
=
\operatorname{concat}_{\ell,b}
\Delta T_b^{(\ell)}(v)
$$

The final `score_transfer` is a robust anomaly score of this vector within the node's observed class and approximate degree regime.

The conditioning removes differences that are normal for a class or degree range. It does not assume a fixed direction such as “poisoned nodes always have more high-frequency energy.”

## 6. H1: local structural spectral moments

For each node $v$, define

$$
M_k(v)=e_v^\top L^ke_v
$$

GSDD-v0.1 estimates $\operatorname{diag}(L^k)$ using Hutchinson probes:

$$
\operatorname{diag}(L^k)
=
\mathbb E_z\left[z\odot L^kz\right]
$$

where each entry of $z$ is independently sampled from $\{-1,+1\}$.

The default orders are

$$
k\in\{2,3,4,5\}
$$

These moments are auxiliary features. They are not the principal claimed novelty.

## 7. H2: input spectral anomaly

The input-feature statistic is simply

$$
s_X(v)=p(v;X)
$$

Its robust deviation from the class- and degree-conditioned reference distribution gives `score_input_spectrum`.

This can detect obvious feature or structural-feature triggers, but a distribution-preserving trigger may evade it. H4 is intended to remain informative when the raw input itself is not an obvious outlier.

## 8. Robust calibration

Let $s_j(v)$ be one coordinate of a diagnostic feature. In a reference group $\mathcal C(v)$, define

$$
\widehat\mu_j(v)
=
\operatorname{median}_{u\in\mathcal C(v)}s_j(u)
$$

$$
\widehat\sigma_j(v)
=
1.4826\operatorname{median}_{u\in\mathcal C(v)}
\left|s_j(u)-\widehat\mu_j(v)\right|+\varepsilon
$$

and

$$
z_j(v)
=
\min
\left\{
\frac{|s_j(v)-\widehat\mu_j(v)|}{\widehat\sigma_j(v)},
c
\right\}
$$

Reference groups first use observed class and degree quantile. If the group is too small, calibration falls back to the observed class.

The component score is the mean of the two largest coordinate-wise robust deviations. This is less brittle than a single maximum and less dilutive than averaging every coordinate.

## 9. Diagnostic fusion

The five component scores are

$$
S_A(v)
$$

$$
S_X(v)
$$

$$
S_{\mathrm{model}}(v)
$$

$$
S_{\mathrm{relation}}(v)
$$

$$
S_{\mathrm{transfer}}(v)
$$

They are robustly standardized again, then averaged:

$$
S_{\mathrm{combined}}(v)
=
\frac15
\sum_{r\in\{A,X,\mathrm{model},\mathrm{relation},\mathrm{transfer}\}}
\widetilde S_r(v)
$$

This fusion is intentionally simple. It prevents a learned detector from concealing whether H3 or H4 actually works.

## 10. Falsification criteria

The main hypothesis should be considered unsupported in a setting when all of the following hold across several seeds:

1. Triggered-test ASR is high
2. `score_transfer` remains near chance
3. `score_model_js` remains near chance
4. Only structure or raw input scores separate poisoned nodes

That outcome would mean the experiment is detecting obvious trigger artifacts rather than a distinct cross-view spectral-learning phenomenon.
