# EE5609 Project 10: Statistical Accumulator Sizing

How narrow can the accumulator of an int8 dot product be if a tiny overflow
probability (1e-9) is acceptable, compared with the worst-case width?

## Folder layout

```
ee5609/
├── run_all.sh                     reproduce everything (bash run_all.sh)
├── montecarlo/
│   └── week1_montecarlo.py        Week 1: synthetic K=32 Monte Carlo vs Gaussian/Bernstein/Hoeffding
├── model/
│   ├── common.py                  paths, MNIST loader, bit helpers, exact integer MLP
│   ├── week2_train_quantize.py    Week 2a: train MLP 784-256-128-10 (NumPy), quantize to int8
│   ├── week2_accumulators.py      Week 2b: log accumulators, w^T Sigma w, bound ladder, figures
│   ├── mlp_float.npz              trained float weights
│   └── mlp_int8.npz               quantized model (int8 weights + scales)
├── data/mnist/                    the four MNIST .gz files (downloaded automatically if missing)
├── results/
│   ├── week1/                     overflow-vs-width plot, CSV, log (2M trials)
│   └── week2/                     figures (PNG + PDF), CSV tables, summary.md, acc_test.npz
├── rtl/  synth/                   Week 4 (Verilog MAC, Vivado/Yosys)
└── report/                        LaTeX report
```

## Fixed design choices

| Item | Choice | Why |
|---|---|---|
| Weights | int8, narrow range [-127, 127], symmetric, one scale per output neuron | standard per-channel PTQ |
| Activations | uint8 [0, 255] | MNIST pixels are already uint8; ReLU outputs are non-negative |
| Accumulator | holds only the integer dot product w^T x | bias and scale applied after the MAC, as in SIRA |
| Datatype bound | K * 127 * 255 | K = 32 gives 21 bits, the same as Quiz 2 |
| Tiles | each dot product also analysed as 32-term partial sums | links the real network to Quiz 2 / Week 1 |
| Statistics | mean and Sigma from 60k training images, everything checked on the 10k test images | out-of-sample validation |
| Statistical width | min(Gaussian width, SIRA width) | the true sum can never leave the SIRA interval |

## Run

```bash
cd ~/ee5609
source venv/bin/activate
bash run_all.sh
```

Or one step at a time:

```bash
python montecarlo/week1_montecarlo.py --out results/week1   # ~10 s
python model/week2_train_quantize.py                         # ~20 s
python model/week2_accumulators.py                           # ~10 s
```

The model is trained with plain NumPy and a fixed seed. On another machine,
BLAS rounding can change the last digits of the trained weights, so the numbers
can move slightly, but the conclusions do not.

## Figures for the report (results/)

| File | Shows |
|---|---|
| week1/week1_overflow_vs_width.png | Week 1: synthetic Monte Carlo vs three tail formulas |
| week2/fig1_bound_ladder | bits per layer: datatype, SIRA, Gaussian 1e-15, Gaussian 1e-9, measured |
| week2/fig2_overflow_vs_width | Week 1 plot repeated on real MNIST accumulators |
| week2/fig3_tail_vs_gaussian | real accumulator tails vs the Gaussian |
| week2/fig4_per_neuron_widths | each neuron's own width; the layer is set by its worst neuron |
| week2/fig5_quadratic_form | eigenvalues of Sigma; w^T Sigma w vs diagonal and lambda_max estimates |
| week2/fig6_accumulator_histograms | every test accumulator value with register limits |
