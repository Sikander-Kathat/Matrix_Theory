# Milestone 2 findings (Tasks 1-3)

All numbers come from `model/week3_validation.py` (about 90 s to run). Tables are in `summary.md`, figures are `fig1`-`fig4`.

**How to read the numbers.** *Predicted* means a model output: the Gaussian model uses the mean wᵀμ and the variance wᵀΣw, both estimated from the 60,000 training images. *Observed* means a count from bit-true inference on the 10,000 test images, given with an exact 95% interval. When nothing is observed, we only report an upper bound (about 3/n). No probability smaller than the measurement floor is claimed as observed.

## Setup

The network is the same MNIST MLP (784-256-128-10), int8 weights and uint8 activations.

The datapath is now bit-true fixed point, with no floating point anywhere:

- an int8 × uint8 MAC into a B-bit accumulator, with either wrap-around or saturation after every add;
- the bias added in a 32-bit adder after the accumulator;
- integer rescaling: a 31-bit multiplier, a right shift, round-half-up;
- ReLU and a clip to 0-255.

Wrap and saturation were checked against a term-by-term simulation.

The full-width fixed-point network scores 98.30%. It gives the same predictions as the earlier floating-point-rescale model on all 10,000 test images, with 0 mismatches.

## Widths per layer (bits)

| Layer | K | Datatype | SIRA | Statistical (Gaussian, predicted 1e-9) | Largest observed |
|---|---|---|---|---|---|
| fc1 | 784 | 26 | 23 | 22 (SIRA-1) | 22 |
| fc2 | 256 | 24 | 22 | 20 (SIRA-2) | 19 |
| fc3 | 128 | 23 | 21 | 20 (SIRA-1) | 19 |

## Task 1: predicted vs observed overflow

The table counts test-set dot products; MAC operations = dot products × K.

| Layer | Width | Dot products | MAC operations | Predicted count | Observed overflows |
|---|---|---|---|---|---|
| fc1 | SIRA-2 (21) | 2,560,000 | 2.0 × 10⁹ | 53.6 | **58** (rate 2.3e-5, 95% CI 1.7e-5 to 2.9e-5) |
| fc1 | SIRA-1 (22) | 2,560,000 | 2.0 × 10⁹ | 6 × 10⁻⁸ | 0 (rate below 1.2e-6) |
| fc2 | SIRA-2 (20) | 1,280,000 | 3.3 × 10⁸ | ≈ 0 | 0 (rate below 2.3e-6) |
| fc3 | SIRA-2 (19) | 100,000 | 1.3 × 10⁷ | 3.35 | 0 (rate below 3.0e-5) |

- **Where overflow is frequent enough to measure, the Gaussian prediction matches.** In fc1 at 21 bits it predicted 53.6 and we observed 58. All 58 were negative sums from 12 neurons, at 2.7σ to 5.8σ from the mean (median 3.7σ).
- **At the statistical widths, nothing overflowed on the test set.** That only shows the rate is below about 1e-6. It does **not** confirm 1e-9: the 1e-9 figure remains a model prediction.
- **At the SIRA width, overflow is impossible by construction**, and we confirmed 0.

## Task 2: accuracy with the bit-true datapath (10,000 test images)

| Configuration | Widths fc1/fc2/fc3 | Accuracy | Mismatches vs full width |
|---|---|---|---|
| Full width (baseline) | 32/32/32 | 98.30% | 0 |
| Datatype bound | 26/24/23 | 98.30% | 0 |
| SIRA | 23/22/21 | 98.30% | 0 |
| **Statistical** | **22/20/20** | **98.30%** | **0** |
| SIRA-1, all layers | 22/21/20 | 98.30% | 0 |
| SIRA-2, all layers | 21/20/19 | 98.30% | 0 (58 overflows in fc1, none changed a class) |

Wrap-around and saturation give identical results for every configuration in this table.

**Layer sensitivity** (one layer narrowed at a time, wrap-around):

- **fc1 overflows first.** Its largest observed value is only 1 bit below SIRA, and at SIRA-3 (20 bits) there are 289 mismatches.
- **fc3, the output layer, is the most damaging.** At SIRA-3 almost every overflow flips a class: 2,182 mismatches from 2,638 overflows.
- **fc2 is the most tolerant.** At SIRA-4 there are only 4 mismatches.

Saturation is far more forgiving than wrap-around. At SIRA-3, fc1 has 0 mismatches instead of 289, and fc3 has 0 instead of 2,182. The reason is that a saturated value keeps its sign, while a wrapped value flips sign.

Rounding matters slightly: a truncating rescale instead of rounding gives 13 mismatches at full width.

## Task 3: how reliable is the Gaussian model?

**Synthetic 32-term dot products** (independent, bounded terms, so Bernstein's assumptions hold; 2 million trials):

| Distribution | Gaussian at the tail | Evidence |
|---|---|---|
| Uniform int8 | consistent (accurate) | 18 bits: observed 2.0e-5, Gaussian 2.2e-5 |
| Gaussian weights + ReLU | **optimistic** | 16 bits: observed 1.0e-6, Gaussian 9.9e-10 (about 1000×) |
| Laplacian weights + ReLU | **optimistic** | 15 bits: observed 5.4e-4, Gaussian 1.5e-5 (36×) |

The Bernstein bound was never below an observed rate. It is safe, but costs 1 to 3 more bits: at 1e-9 it gives 19 to 20 bits, against the Gaussian's 16 to 19.

**MNIST layers.** The terms are correlated: the off-diagonal part of Σ is most of the variance. So Bernstein's independence assumption fails, and we did not apply it. Instead we compared the Gaussian model directly with the empirical distribution of standardized accumulator values:

- **fc1 and fc2:** close to Gaussian up to about 3σ, then increasingly **optimistic**. The real tail is 3.8× heavier at 4σ in fc1 and 18× at 5σ; fc2 is 1.7× at 4σ and 8× at 5σ. fc1's left (negative) tail is the heavy side.
- **fc3:** clearly not Gaussian. It has two peaks, because one logit is high for the true class. The Gaussian is **optimistic** from about 2σ (4.5× at 3σ).
- **Cantelli** (a bound that needs only the mean and variance, valid without independence) is always safe, but at 1e-9 it gives back the SIRA width. So without a shape assumption we cannot go below SIRA at very small target probabilities.

## Limitations of Gaussian-based sizing, and proposed improvements

1. **The tails are heavier than Gaussian.** The model under-predicts rare overflows: 4× to 18× at 4-5σ for MNIST, up to 1000× for synthetic ReLU inputs. Its 1e-9 widths are therefore optimistic estimates, not guarantees.
2. **Rare events cannot be validated by sampling.** With 10⁶ to 10⁷ dot products we can only check down to about 1e-6, so the claimed safety margin rests on the model.
3. **There is no rigorous bound for correlated inputs.** Bernstein needs independence, and Cantelli is too loose to beat SIRA.

**Proposed improvements**, all keeping the existing network:

- **(a)** Add a safety margin where the measured headroom is small. In fc1 the statistical width (22 bits) equals the largest observed width, so fc1 has no spare bit; fc2 and fc3 have 1 bit each.
- **(b)** Use **saturating accumulators**. On this network they turn rare overflows into harmless errors.
- **(c)** Replace the Gaussian with a tail model fitted to the measured data, such as a heavier-tailed (Student-t or generalized Pareto) fit to the standardized values.
- **(d)** Derive a bound for correlated inputs, for example by decorrelating with the eigenvectors of Σ before applying Bernstein.
