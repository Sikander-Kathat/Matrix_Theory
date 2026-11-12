# Week 2 results: real accumulators of the int8 MNIST MLP

Int8 test accuracy: 0.9833. Widths in bits (two's complement), layer width = worst neuron.

## Full dot product

| layer | K | datatype | SIRA l1 | Gauss 1e-15/dot | Gauss 1e-9/dot | Gauss 1e-9/inference | largest seen (test) |
|---|---|---|---|---|---|---|---|
| fc1 | 784 | 26 | 24 | 23 | 22 | 22 | 22 |
| fc2 | 256 | 24 | 22 | 20 | 20 | 20 | 19 |
| fc3 | 128 | 23 | 21 | 21 | 20 | 20 | 19 |

## K = 32 tiles (Quiz 2 setting)

| layer | tiles | datatype | SIRA l1 | Gauss 1e-15/dot | Gauss 1e-9/dot | largest seen (test) |
|---|---|---|---|---|---|---|
| fc1 | 25 | 21 | 20 | 20 | 20 | 20 |
| fc2 | 8 | 21 | 20 | 19 | 18 | 18 |
| fc3 | 4 | 21 | 20 | 19 | 19 | 18 |

## Quadratic-form diagnostics

| layer | max rel. error w^T Sigma w vs measured variance | lambda_max share of trace | median sqrt(diag)/true std | median sqrt(lambda_max)*||w||/true std | median SIRA half-range / std | median excess kurtosis |
|---|---|---|---|---|---|---|
| fc1 | 9.5e-14 | 0.10 | 0.47 | 4.00 | 21.0 | 0.21 |
| fc2 | 3.8e-14 | 0.12 | 0.60 | 3.30 | 38.2 | -0.08 |
| fc3 | 1.3e-14 | 0.20 | 0.38 | 1.92 | 13.7 | 1.31 |
