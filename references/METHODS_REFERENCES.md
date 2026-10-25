# Method-component references

Keys refer to `references/references.bib`. Bibliographic fields were checked via Crossref / arXiv APIs; entries with remaining doubts carry a `note` containing UNVERIFIED. PDFs (open access only) are in `references/papers/`.

| Pipeline component | Where in code | Reference keys | Why |
|---|---|---|---|
| Relaxed readout (LASSO select, then less-shrunk / OLS refit) and warm start | src/hyperdiscover/train/alternating.py | meinshausen2007relaxed, belloni2013least, tibshirani1996regression | Removes LASSO shrinkage bias after support selection |
| Non-negative least squares refit | src/hyperdiscover/train/alternating.py | lawson1995solving | NNLS active-set algorithm (original 1974 book; SIAM 1995 reprint cited) |
| BIC with LASSO degrees of freedom | src/hyperdiscover/distill/select.py | schwarz1978estimating, zou2007degrees, efron2004least | BIC criterion; df of lasso = active-set size (Zou et al.) justifies k in BIC |
| Effective sample size under autocorrelation, n(1-rho)/(1+rho) | src/hyperdiscover/distill/select.py | bayley1946effective | Autocorrelated samples carry fewer independent observations; BIC uses n_eff |
| Backward elimination / best-subset style search | src/hyperdiscover/distill/select.py | bertsimas2016best | Modern reference for exact/greedy subset selection |
| Library beam search; STLSQ thresholding; ensemble SINDy | src/hyperdiscover/distill/pysr_runner.py | bertsimas2016best, brunton2016discovering, fasel2022ensemble, kaptanoglu2022pysindy | Sparse-regression term selection on candidate library |
| PySR / SR benchmarking context | src/hyperdiscover/distill/pysr_runner.py | cranmer2023interpretable, lacava2021contemporary | Symbolic regression engine; standard SR benchmark (SRBench) |
| Hard-concrete L0 gates | src/hyperdiscover/model/gates.py | louizos2018learning, maddison2017concrete, jang2017categorical | Hard-concrete builds on the Concrete / Gumbel-softmax relaxation |
| GNN-encoder gates (amortized structure inference, message passing) | src/hyperdiscover/model/gnn_gates.py | gilmer2017neural, battaglia2018relational, kipf2018neural, lowe2022amortized, zhu2021survey, feng2019hypergraph, bai2021hypergraph | MPNN / graph-network framework; amortized edge inference (NRI, ACD); structure-learning survey; hypergraph NNs for triple-level encoders |
| Joint kicks: Richardson extrapolation | src/hyperdiscover/intervene/joint_kick.py | richardson1911approximate, richardson1927deferred | Readouts at τ and 2τ, combined as 2·x(τ) − x(2τ)/2, cancel the O(τ²) term of each response difference (indirect paths j → l → i) |
| Joint kicks: chi-squared tests with Bonferroni correction (BH as alternative) | src/hyperdiscover/intervene/joint_kick.py | dunn1961multiple, benjamini1995controlling | Family-wise error control over many candidate triples; FDR alternative |
| Identifiability fraction (pure-interaction share; backfitting) | src/hyperdiscover/eval/identifiability.py | hooker2007generalized, lengerich2020purifying, buja1989linear, hastie1990generalized | fANOVA decomposition with dependent inputs; purification; backfitting for additive main effects |
| Derivative estimation (Savitzky-Golay, TV-regularized) | src/hyperdiscover/data/derivatives.py | savitzky1964smoothing, chartrand2011numerical | Standard smoothing differentiators for noisy series |
| Weak-form loss | src/hyperdiscover/data/weak_form.py | messenger2021weak, reinbold2020using | Integrate against test functions, avoiding differentiation |
| NRI baseline | src/hyperdiscover/baselines/nri.py | kipf2018neural, gilmer2017neural | Neural relational inference; encoder/decoder message passing |

## Status
- Open-access PDFs downloaded to `references/papers/`: gilmer2017_mpnn, battaglia2018_graph-networks, zhu2021_graph-structure-learning, feng2019_hgnn, bai2021_hypergraph-convolution, maddison2017_concrete, jang2017_gumbel-softmax, lacava2021_srbench, belloni2013_post-lasso-ols, zou2007_lasso-dof, efron2004_lars, bertsimas2016_best-subset.
- Not downloaded (publisher-hosted or books; no open-access copy sought or located in this pass): Meinshausen 2007, Schwarz 1978, Bayley & Hammersley 1946, Richardson 1911/1927, Savitzky & Golay 1964, Chartrand 2011, Hooker 2007, Buja et al. 1989, Dunn 1961, Benjamini & Hochberg 1995, Tibshirani 1996, books (Lawson & Hanson; Hastie & Tibshirani). Their bibliographic records were verified through Crossref.
- UNVERIFIED details: see `note` fields (Gilmer 5th author/venue, Feng 5th author, Maddison/Jang/La Cava venues, Zhu journal version, Hastie & Tibshirani book publisher, Lawson & Hanson 1974 original edition, exact n_eff expression inside Bayley & Hammersley).
- Component → file mapping checked against the code (2026-10-26); every listed file exists and implements the listed component.
