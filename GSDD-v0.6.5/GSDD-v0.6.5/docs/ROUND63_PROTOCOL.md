# Round 6.3 Protocol: Clean-label Factorial Audit

## 1. Motivation

For a clean-label attack, selected poisoned training nodes already belong to the target class and their labels are not changed. Therefore the old dirty-label conditions

$$
\mathcal D_{\mathrm{full}}
\quad\text{and}\quad
\mathcal D_{\mathrm{trigger-only}}
$$

are identical. Any validity rule based on

$$
\operatorname{ASR}_{\mathrm{full}}
-
\operatorname{ASR}_{\mathrm{trigger-only}}
$$

must equal zero and cannot audit clean-label binding.

## 2. Training factor

Train two GCNs from the same initial state and the same training RNG seed:

$$
M_c=\operatorname{Train}(G)
$$

$$
M_p=\operatorname{Train}(G\oplus T_{\mathrm{train}})
$$

All selected victims already have label $y_t$, so no labels are modified.

## 3. Test-trigger factor

Each model is evaluated under:

$$
\varnothing,
\quad
T,
\quad
T_{\mathrm{shuffle}}
$$

The shuffled trigger is produced by a cyclic permutation over victim assignments:

$$
T_{\mathrm{shuffle},i}=T_{\pi(i)}
$$

where $\pi$ has no fixed points when at least two victims exist. Consequently,

$$
\{T_i\}_{i=1}^m
=
\{T_{\mathrm{shuffle},i}\}_{i=1}^m
$$

as multisets. The control preserves marginal trigger distribution and topology while breaking context-specific matching.

## 4. Functional validity

Define

$$
A_{p,m}=\operatorname{ASR}(M_p,T)
$$

and

$$
A_{\mathrm{ctrl}}
=
\max\left\{
\operatorname{ASR}(M_c,\varnothing),
\operatorname{ASR}(M_c,T),
\operatorname{ASR}(M_p,\varnothing),
\operatorname{ASR}(M_p,T_{\mathrm{shuffle}})
\right\}
$$

Then

$$
\Delta_{\mathrm{CL}}=A_{p,m}-A_{\mathrm{ctrl}}
$$

A run is admitted only if

$$
A_{p,m}\geq0.80
$$

$$
\operatorname{ASR}(M_c,\varnothing)\leq0.10
$$

$$
\operatorname{ASR}(M_c,T)\leq0.20
$$

$$
\operatorname{ASR}(M_p,\varnothing)\leq0.20
$$

$$
\operatorname{ASR}(M_p,T_{\mathrm{shuffle}})\leq0.20
$$

$$
\Delta_{\mathrm{CL}}\geq0.40
$$

## 5. Spectral interaction

For model $M$ and graph-trigger condition $Q$, let $S^{(\ell)}(M,Q,v)$ denote the multiband log-gain vector at node $v$ and layer $\ell$.

The primary clean-label interaction is

$$
D_{\mathrm{CL}}^{(\ell)}(v)
=
\left[
S^{(\ell)}(M_p,T,v)-S^{(\ell)}(M_c,T,v)
\right]
-
\left[
S^{(\ell)}(M_p,T_{\mathrm{shuffle}},v)
-
S^{(\ell)}(M_c,T_{\mathrm{shuffle}},v)
\right]
$$

This removes:

- generic trigger input energy
- generic clean-versus-poisoned model differences
- marginal trigger-distribution differences

while retaining context-specific learned binding.

## 6. Candidate population

Node-level detection is evaluated only among labeled training nodes satisfying

$$
y_v=y_t
$$

This prevents class membership from trivially separating poisoned and clean nodes. Cora has 20 labeled training nodes per class; poison counts are restricted to 4, 8, and 12 to preserve same-class controls.

## 7. Target scan

Before generator training, one clean GCN estimates for every class $c$:

$$
B_c
=
\Pr\left(\widehat y=c\mid y\neq c,\ v\in V_{\mathrm{test}}\right)
$$

At most two classes with

$$
B_c\leq0.15
$$

are retained. This avoids repeating the Round 6.2 failure where target class 0 had a naturally high ASR.

## 8. Required reports

Each run outputs:

- `clean_label_factorial_behavior.csv`
- `clean_label_attack_validity.json`
- `clean_label_factorial_node_scores.csv`
- `clean_label_factorial_raw_features.csv`
- `roc_clean_label_factorial.png`
- `pr_clean_label_factorial.png`
- `clean_label_band_profile_layer*.png`
- `SUMMARY.md`
- `summary.json`

The aggregate outputs:

- `CLEAN_LABEL_FACTORIAL_SUMMARY.md`
- `clean_label_factorial_runs.csv`
- `clean_label_factorial_group_stats.csv`
- `clean_label_factorial_success_criteria.csv`
- `selected_candidates.json`
