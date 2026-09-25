# Missing Data & Statistical Inference

A simulation-based study of how missing-data mechanisms and missing-data handling methods affect regression estimation and statistical inference.

## Project question

**Within a given missingness mechanism, how do different methods for handling missing data change the estimated regression relationship, its uncertainty, and the resulting inference?**

The project treats **MCAR, MAR and MNAR as experimental conditions** and compares seven handling approaches within each condition.

## Model

The complete data were generated from

$$
Y = 5 + 2X_1 - 1.5X_2 + X_3 + \epsilon
$$

with true parameters

$$
\beta_0 = 5, \qquad \beta_1 = 2, \qquad \beta_2 = -1.5, \qquad \beta_3 = 1.
$$

Only the response variable $Y$ was made incomplete; $X_1$, $X_2$ and $X_3$ remained fully observed.

## Missingness mechanisms

- **MCAR:** every observation was given the same missingness probability.
- **MAR:** missingness probability depended on the observed predictor $X_1$.
- **MNAR:** missingness probability depended directly on the original outcome $Y$.

The same incomplete dataset for a given mechanism was supplied to all seven methods.

## Methods compared

1. Complete-case analysis
2. Mean imputation
3. Hot-deck imputation
4. Regression imputation
5. Stochastic regression imputation
6. MICE using predictive mean matching (PMM)
7. EM algorithm

## Evaluation

The primary parameter of interest is $\beta_1$, with true value 2.

The analysis reports:

- point estimate $\hat\beta_1$
- bias $\hat\beta_1 - 2$
- bootstrap standard error
- bootstrap 95% confidence interval

MICE uses multiple imputations and Rubin's rules for pooled inference.

## Main findings from the simulated dataset

The strongest and most consistent numerical pattern is the effect of **mean imputation**. Its $\beta_1$ estimates were substantially below 2 under all three mechanisms:

| Mechanism | Mean-imputation $\hat\beta_1$ | Bias |
|---|---:|---:|
| MCAR | 1.089 | -0.911 |
| MAR | 1.307 | -0.693 |
| MNAR | 1.413 | -0.587 |

Within the same MCAR, MAR and MNAR datasets, the other methods produced estimates clustered much closer to the true value in this particular simulation.

For example, under MAR, the MICE pooled estimate was

$$
\hat\beta_1 = 1.797,
$$

with pooled standard error

$$
SE = 0.094,
$$

and 95% interval

$$
(1.612,\ 1.982).
$$

Under MNAR, MICE produced

$$
\hat\beta_1 = 1.722,
$$

with pooled standard error

$$
SE = 0.108,
$$

and 95% interval

$$
(1.510,\ 1.933).
$$

These are results for **one simulated dataset**, not universal rankings of missing-data methods.

## Important scope note

The project stops at bootstrap inference. It does **not** perform a repeated Monte Carlo study, so it does not estimate long-run empirical bias or confidence-interval coverage across independent datasets.

The MICE implementation is intentionally simplified because only one variable ($Y$) is incomplete. MNAR is treated as a stress test; solving MNAR generally requires additional assumptions or sensitivity analysis.

## Repository structure

```text
missing-data-stat-inference/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── missing_data_statistical_inference.ipynb
├── report/
│   ├── Missing_Data_Statistical_Inference_Final_Report.pdf
│   └── Missing_Data_Statistical_Inference_Final_Report.tex
├── figures/
└── results/
```

## Tools

Python • NumPy • pandas • SciPy • statsmodels • Matplotlib • Jupyter • LaTeX
