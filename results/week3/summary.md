# Milestone 2: validation of statistical accumulator sizing (Tasks 1-3)

Predicted = model output (Gaussian from w^T mu and w^T Sigma w on the 60k training images; Bernstein / Cantelli bounds). Observed = counts from bit-true inference on the 10k test images, with exact 95% (Clopper-Pearson) intervals. Zero observed events only give an upper bound (about 3/n); no smaller probability is claimed as observed.

## Table 1. Accumulator widths per layer (bits)

| Layer | K | Datatype | SIRA | SIRA-1 | SIRA-2 | Gaussian model (pred. 1e-9) | Cantelli (pred. 1e-9) | Largest observed |
|---|---|---|---|---|---|---|---|---|
| fc1 | 784 | 26 | 24 | 23 | 22 | 22 | 24 | 22 |
| fc2 | 256 | 24 | 22 | 21 | 20 | 20 | 22 | 19 |
| fc3 | 128 | 23 | 21 | 20 | 19 | 20 | 21 | 19 |

## Table 2. Task 1: predicted vs observed overflow (test set)

| Layer | Config | Bits | Dot products | MAC ops | Predicted P/dot | Predicted count | Observed overflows | Observed rate [95% CI] | Partial-sum overflows | Images affected | Gaussian is |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fc1 | SIRA | 24 | 2560000 | 2007040000 | 0 | 0 | 0 | 0 (95% upper bound 1.2e-06) | 0 | 0 | overflow impossible (SIRA guarantee) |
| fc1 | SIRA-1 | 23 | 2560000 | 2007040000 | 1.49e-154 | 3.81e-148 | 0 | 0 (95% upper bound 1.2e-06) | 0 | 0 | not testable here (prediction below measurement floor) |
| fc1 | SIRA-2 | 22 | 2560000 | 2007040000 | 5.30e-14 | 1.36e-07 | 0 | 0 (95% upper bound 1.2e-06) | 0 | 0 | not testable here (prediction below measurement floor) |
| fc2 | SIRA | 22 | 1280000 | 327680000 | 0 | 0 | 0 | 0 (95% upper bound 2.3e-06) | 0 | 0 | overflow impossible (SIRA guarantee) |
| fc2 | SIRA-1 | 21 | 1280000 | 327680000 | 1.26e-109 | 1.61e-103 | 0 | 0 (95% upper bound 2.3e-06) | 0 | 0 | not testable here (prediction below measurement floor) |
| fc2 | SIRA-2 | 20 | 1280000 | 327680000 | 2.69e-30 | 3.45e-24 | 0 | 0 (95% upper bound 2.3e-06) | 0 | 0 | not testable here (prediction below measurement floor) |
| fc3 | SIRA | 21 | 100000 | 12800000 | 0 | 0 | 0 | 0 (95% upper bound 3.0e-05) | 0 | 0 | overflow impossible (SIRA guarantee) |
| fc3 | SIRA-1 | 20 | 100000 | 12800000 | 1.72e-16 | 1.72e-11 | 0 | 0 (95% upper bound 3.0e-05) | 0 | 0 | not testable here (prediction below measurement floor) |
| fc3 | SIRA-2 | 19 | 100000 | 12800000 | 2.50e-05 | 2.5 | 0 | 0 (95% upper bound 3.0e-05) | 0 | 0 | consistent |

## Table 3. Task 2: accuracy, bit-true fixed point (10,000 test images)

| Configuration | Widths fc1/fc2/fc3 | Overflow mode | Accuracy | Mismatches vs baseline | Final-sum overflows fc1/fc2/fc3 |
|---|---|---|---|---|---|
| Float rescaling (Week 2 reference) | 32/32/32 | - | 0.9833 | 0 | - |
| Fixed-point, full width (baseline) | 32/32/32 | - | 0.9833 | 0 | 0/0/0 |
| Fixed-point, truncating rescale | 32/32/32 | - | 0.9832 | 2 | 0/0/0 |
| Datatype bound | 26/24/23 | wrap | 0.9833 | 0 | 0/0/0 |
| Datatype bound | 26/24/23 | saturate | 0.9833 | 0 | 0/0/0 |
| SIRA | 24/22/21 | wrap | 0.9833 | 0 | 0/0/0 |
| SIRA | 24/22/21 | saturate | 0.9833 | 0 | 0/0/0 |
| Statistical (Gaussian model 1e-9) | 22/20/20 | wrap | 0.9833 | 0 | 0/0/0 |
| Statistical (Gaussian model 1e-9) | 22/20/20 | saturate | 0.9833 | 0 | 0/0/0 |
| SIRA-1 (all layers) | 23/21/20 | wrap | 0.9833 | 0 | 0/0/0 |
| SIRA-1 (all layers) | 23/21/20 | saturate | 0.9833 | 0 | 0/0/0 |
| SIRA-2 (all layers) | 22/20/19 | wrap | 0.9833 | 0 | 0/0/0 |
| SIRA-2 (all layers) | 22/20/19 | saturate | 0.9833 | 0 | 0/0/0 |

## Table 4. Task 2: layer sensitivity (one layer narrowed, others full width)

| Layer | Bits | Below SIRA | Mode | Accuracy | Mismatches | Overflows |
|---|---|---|---|---|---|---|
| fc1 | 24 | 0 | wrap | 0.9833 | 0 | 0 |
| fc1 | 24 | 0 | saturate | 0.9833 | 0 | 0 |
| fc1 | 23 | 1 | wrap | 0.9833 | 0 | 0 |
| fc1 | 23 | 1 | saturate | 0.9833 | 0 | 0 |
| fc1 | 22 | 2 | wrap | 0.9833 | 0 | 0 |
| fc1 | 22 | 2 | saturate | 0.9833 | 0 | 0 |
| fc1 | 21 | 3 | wrap | 0.9833 | 0 | 65 |
| fc1 | 21 | 3 | saturate | 0.9833 | 0 | 65 |
| fc1 | 20 | 4 | wrap | 0.9577 | 307 | 27413 |
| fc1 | 20 | 4 | saturate | 0.9832 | 1 | 27413 |
| fc1 | 19 | 5 | wrap | 0.7210 | 2770 | 340180 |
| fc1 | 19 | 5 | saturate | 0.9822 | 38 | 340180 |
| fc1 | 18 | 6 | wrap | 0.2323 | 7675 | 1047424 |
| fc1 | 18 | 6 | saturate | 0.9148 | 798 | 1047424 |
| fc2 | 22 | 0 | wrap | 0.9833 | 0 | 0 |
| fc2 | 22 | 0 | saturate | 0.9833 | 0 | 0 |
| fc2 | 21 | 1 | wrap | 0.9833 | 0 | 0 |
| fc2 | 21 | 1 | saturate | 0.9833 | 0 | 0 |
| fc2 | 20 | 2 | wrap | 0.9833 | 0 | 0 |
| fc2 | 20 | 2 | saturate | 0.9833 | 0 | 0 |
| fc2 | 19 | 3 | wrap | 0.9833 | 0 | 0 |
| fc2 | 19 | 3 | saturate | 0.9833 | 0 | 0 |
| fc2 | 18 | 4 | wrap | 0.9832 | 1 | 2098 |
| fc2 | 18 | 4 | saturate | 0.9833 | 0 | 2098 |
| fc2 | 17 | 5 | wrap | 0.9346 | 544 | 121292 |
| fc2 | 17 | 5 | saturate | 0.9831 | 11 | 121292 |
| fc2 | 16 | 6 | wrap | 0.2528 | 7452 | 487239 |
| fc2 | 16 | 6 | saturate | 0.9801 | 76 | 487239 |
| fc3 | 21 | 0 | wrap | 0.9833 | 0 | 0 |
| fc3 | 21 | 0 | saturate | 0.9833 | 0 | 0 |
| fc3 | 20 | 1 | wrap | 0.9833 | 0 | 0 |
| fc3 | 20 | 1 | saturate | 0.9833 | 0 | 0 |
| fc3 | 19 | 2 | wrap | 0.9833 | 0 | 0 |
| fc3 | 19 | 2 | saturate | 0.9833 | 0 | 0 |
| fc3 | 18 | 3 | wrap | 0.8106 | 1728 | 2110 |
| fc3 | 18 | 3 | saturate | 0.9833 | 0 | 2110 |
| fc3 | 17 | 4 | wrap | 0.0916 | 9012 | 25875 |
| fc3 | 17 | 4 | saturate | 0.9820 | 33 | 25875 |
| fc3 | 16 | 5 | wrap | 0.0955 | 9035 | 57352 |
| fc3 | 16 | 5 | saturate | 0.9368 | 553 | 57352 |
| fc3 | 15 | 6 | wrap | 0.0976 | 9021 | 77133 |
| fc3 | 15 | 6 | saturate | 0.8121 | 1841 | 77133 |

## Table 5. Task 3: synthetic distributions, Gaussian vs observed (2M trials)

| Distribution | Bits | Observed rate [95% CI] | Gaussian pred. | Bernstein bound | Observed / Gaussian | Gaussian is |
|---|---|---|---|---|---|---|
| Uniform int8 (pessimistic) | 17 | 3.41e-02 [3.4e-02, 3.4e-02] | 3.39e-02 | 3.89e-01 | 1.0x | consistent |
| Uniform int8 (pessimistic) | 18 | 2.00e-05 [1.4e-05, 2.7e-05] | 2.21e-05 | 1.17e-02 | 0.9x | consistent |
| Gaussian weights + ReLU | 15 | 4.71e-03 [4.6e-03, 4.8e-03] | 2.25e-03 | 6.38e-01 | 2.1x | optimistic (real rate higher) |
| Gaussian weights + ReLU | 16 | 1.00e-06 [1.2e-07, 3.6e-06] | 9.94e-10 | 1.48e-01 | 1006.2x | optimistic (real rate higher) |
| Laplacian weights + ReLU | 14 | 3.83e-02 [3.8e-02, 3.9e-02] | 3.05e-02 | 1.00e+00 | 1.3x | optimistic (real rate higher) |
| Laplacian weights + ReLU | 15 | 5.44e-04 [5.1e-04, 5.8e-04] | 1.51e-05 | 5.44e-01 | 35.9x | optimistic (real rate higher) |

## Table 6. Task 3: synthetic design widths at 1e-9 (predicted) vs observed maximum

| Distribution | Largest observed (bits) | Gaussian pred. 1e-9 | Bernstein 1e-9 | Worst case | Bernstein never below observed |
|---|---|---|---|---|---|
| Uniform int8 (pessimistic) | 19 | 19 | 20 | 21 | True |
| Gaussian weights + ReLU | 17 | 16 | 19 | 21 | True |
| Laplacian weights + ReLU | 16 | 16 | 19 | 21 | True |

## Table 7. Task 3: MNIST layers, tail beyond k sigma (test set)

| Layer | k (sigma) | Count | Observed rate [95% CI] | Gaussian pred. | Observed / Gaussian | Gaussian is |
|---|---|---|---|---|---|---|
| fc1 | 2 | 124547 | 4.87e-02 [4.8e-02, 4.9e-02] | 4.55e-02 | 1.07x | optimistic (real rate higher) |
| fc1 | 3 | 11345 | 4.43e-03 [4.4e-03, 4.5e-03] | 2.70e-03 | 1.64x | optimistic (real rate higher) |
| fc1 | 4 | 637 | 2.49e-04 [2.3e-04, 2.7e-04] | 6.33e-05 | 3.93x | optimistic (real rate higher) |
| fc1 | 5 | 28 | 1.09e-05 [7.3e-06, 1.6e-05] | 5.73e-07 | 19.08x | optimistic (real rate higher) |
| fc1 | 6 | 0 | 0 (< 1.2e-06) | 1.97e-09 | - | consistent |
| fc2 | 2 | 58701 | 4.59e-02 [4.5e-02, 4.6e-02] | 4.55e-02 | 1.01x | consistent |
| fc2 | 3 | 4075 | 3.18e-03 [3.1e-03, 3.3e-03] | 2.70e-03 | 1.18x | optimistic (real rate higher) |
| fc2 | 4 | 99 | 7.73e-05 [6.3e-05, 9.4e-05] | 6.33e-05 | 1.22x | consistent |
| fc2 | 5 | 3 | 2.34e-06 [4.8e-07, 6.8e-06] | 5.73e-07 | 4.09x | consistent |
| fc2 | 6 | 0 | 0 (< 2.3e-06) | 1.97e-09 | - | consistent |
| fc3 | 2 | 7782 | 7.78e-02 [7.6e-02, 7.9e-02] | 4.55e-02 | 1.71x | optimistic (real rate higher) |
| fc3 | 3 | 1220 | 1.22e-02 [1.2e-02, 1.3e-02] | 2.70e-03 | 4.52x | optimistic (real rate higher) |
| fc3 | 4 | 22 | 2.20e-04 [1.4e-04, 3.3e-04] | 6.33e-05 | 3.47x | optimistic (real rate higher) |
| fc3 | 5 | 0 | 0 (< 3.0e-05) | 5.73e-07 | - | consistent |
| fc3 | 6 | 0 | 0 (< 3.0e-05) | 1.97e-09 | - | consistent |
