# API reference

## `HyperDiscover` (`hyperdiscover.api`)

```python
HyperDiscover(state_dim=1, max_order=3, config="default", periodic=False,
              distill_engine="auto", pysr_options=None, seed=0, verbose=True)
```

| argument | meaning |
|---|---|
| `state_dim` | d; only d = 1 is supported end-to-end |
| `max_order` | 2 = pairwise only, 3 = pairwise + triadic |
| `config` | preset (`"default"`, `"fast"`), YAML path, or dict merged into `pipeline.DEFAULT_CONFIG` |
| `periodic` | inputs are phases: sin/cos embedding, per-node constants c_i on (`node_constants: auto`) |
| `distill_engine` | `"pysr"`, `"library"` (fallback, not symbolic regression), `"auto"`, `"none"` |
| `pysr_options` | overrides `distill.pysr_runner.DEFAULT_PYSR` (operators, maxsize, timeout, restarts …) |

### Methods

| method | what it does |
|---|---|
| `fit(X, t, segment_starts=None, val_fraction=0.2, out_dir=None)` | X: [M, T, N, d] (or [M, T, N]); splits whole trajectories into train/val; runs stages 0–5 |
| `fit_symbolic(X, t, ...)` | stages 4–5 only, on an already fitted or loaded structure |
| `save_structure(path)` / `HyperDiscover.from_structure_checkpoint(path, X, t, ...)` | separate GPU fitting from CPU distillation |
| `simulate(x0, t)` | integrate the discovered symbolic model; x0: [N, d] or [B, N, d] |
| `simulate_neural(x0, t)` | integrate the learned neural model (deterministic gates) |
| `discovered_system()` | the symbolic model as a `SymbolicModel` (has `rhs`, `coupling_parts`, `g_scale`, `h_scale`) |
| `summary()` / `report()` / `save(dir)` | JSON-able summary, Markdown report, files on disk |

### Fitted attributes

| attribute | type |
|---|---|
| `pairwise_edges_` | `[(i, j, score)]`, undirected, score = gate probability (max over directions) |
| `hyperedges_` | `[(i, j, k, score)]`, undirected (max over focal nodes) |
| `equations_` | `{"f", "g", "h"} → sympy` in x_i, x_j, x_k |
| `node_constants_` | `[N, 1]` or `None` |
| `structure_` | `Structure` of the read-out edges / hyperedges |
| `distilled_` | `{kind: DistillResult}` with all candidates and `.pareto()` |
| `selection_` | refit + BIC ranking of candidate combinations |
| `structure_result_` | screened candidates, final gate scores, training histories, timings |

## Configuration keys (`pipeline.DEFAULT_CONFIG`)

| key | default | notes |
|---|---|---|
| `schedule` | `[pairwise, triadic, joint]` | identifiability schedule (a); `[joint]` = all at once |
| `data.deriv_method` | `savgol` | `fd`, `savgol`, `tikhonov`, `tv` |
| `data.weak_kw` | window 21, stride 4, power 4 | weak-form test functions |
| `screen.all_pairs_max_n` / `k_p` | 200 / 30 | every ordered pair below the cut, else top-K_p sources |
| `screen.all_triples_max_n` / `k_t` | 12 / 50 | every focal triple below the cut, else top-K_t per node |
| `model.hidden`, `model.layers` | 64, 3 | MLPs for f, g, h |
| `model.init_prob_pair/tri` | 0.9 / 0.5 | initial P(gate on) |
| `train.<stage>.loss` | `derivative` | `derivative`, `weak`, `mixed` |
| `train.<stage>.lam_pair/lam_tri` | 0.01 / 0.02 | λ_t > λ_p (Occam); unit = fraction of a node's derivative variance |
| `train.<stage>.anova` | 0 | functional-ANOVA pure-interaction penalty weight (option b) |
| `train.joint.rollout_weight/horizon` | 0 / 5 | multiple-shooting loss (option C) |
| `train.<stage>.beta_start/beta_end` | 2/3 → 0.2 (joint) | hard-concrete temperature schedule |
| `train.<stage>.trainable` | per stage | override which parts train |
| `train.prune_between_stages/threshold` | true / 0.05 | drop near-zero candidates between stages |
| `readout.threshold`, `readout.undirected` | 0.5, `max` | structure readout |

## Benchmarks (`hyperdiscover.benchmarks`)

```python
b = benchmarks.load(system_id, N=50, noise=0.0, hyper_kind="open", graph="er", mean_degree=4.0,
                    n_hyper=None, hyper_density=0.3, pairwise_only=False, M=50, T=200, dt=None,
                    seed=0, params=None, kick_std=0.0, kick_every=0, disjoint_hyperedges=False)
b.data.X, b.data.X_clean, b.data.dX_clean, b.data.t, b.structure, b.system, b.truth, b.spec
benchmarks.save(b, path); data, spec, truth = benchmarks.load_file(path); benchmarks.rebuild_system(truth)
```

## Evaluation (`hyperdiscover.eval`)

| function | what it measures |
|---|---|
| `edge_metrics`, `hyperedge_metrics` | AUROC / AUPRC / F1, including unscored candidates; hyperedges also per closure class |
| `false_hyperedge_rate` | predicted hyperedges on pairwise-only systems |
| `equation_report`, `parametric_match` | exact recovery (D-012), coefficient errors, functional nRMSE |
| `combined_coupling_error` | self / pair / tri / summed-coupling / full-RHS errors |
| `forecast_report` | rollout nRMSE vs time, valid prediction time |
| `eval.bifurcation.compare_transitions` | out-of-sample threshold and hysteresis sweep |
| `eval.perturbation.perturbation_test` | interventions (option c) |
| `eval.run_metrics.run_metrics` | all of the above for one run → `metrics.json` |
