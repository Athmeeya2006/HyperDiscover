| method        |   n |   edge_auroc |   edge_f1 |   hyper_auroc |   hyper_f1 |   control_fp_per_run |
|:--------------|----:|-------------:|----------:|--------------:|-----------:|---------------------:|
| sindy_oracle  |  30 |        1     |     0.884 |         1     |      0.972 |                  0   |
| HyperDiscover |  30 |        1     |     0.999 |         0.971 |      0.967 |                  0.3 |
| sindy_generic |  30 |        0.991 |     0.952 |         0.925 |      0.734 |                  3.3 |
| this          |  30 |        0.901 |     0.757 |         0.801 |      0.284 |                 17   |
| arni          |  30 |        0.993 |     0.869 |       nan     |    nan     |                  0   |
| corr          |  30 |        0.801 |     0.495 |       nan     |    nan     |                  0   |
| granger       |  30 |        0.88  |     0.545 |       nan     |    nan     |                  0   |
| pcorr         |  30 |        0.8   |     0.493 |       nan     |    nan     |                  0   |
| te            |  30 |        0.899 |     0.561 |       nan     |    nan     |                  0   |

Hyperedge F1 by kind:

| method        |   closed |   open |
|:--------------|---------:|-------:|
| HyperDiscover |    1     |  0.935 |
| sindy_generic |    0.711 |  0.757 |
| sindy_oracle  |    1     |  0.944 |
| this          |    0.238 |  0.33  |

Paired hyperedge F1, HyperDiscover − baseline (mean diff, Wilcoxon p, n):

- sindy_generic: +0.234, p = 0.0109, n = 20
- sindy_oracle: -0.005, p = 0.219, n = 20
- this: +0.683, p = 0.000179, n = 20
