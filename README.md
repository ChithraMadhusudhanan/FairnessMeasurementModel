# fairmeasure

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Statistically grounded, multi-backend fairness auditing for machine learning.**

**`fairmeasure`** is a unified fairness evaluation toolkit for Python. Rather than requiring teams to navigate disparate APIs, data formats, and conceptual differences between standalone fairness libraries, `fairmeasure` provides a single entry point: `FairnessAnalyzer`.

Under the hood, metrics are routed to established engines (**Fairlearn**, **Aequitas**, or an optimized **native NumPy** implementation). Every reported number is automatically accompanied by **bootstrap confidence intervals**, **standardized effect size ratios**, **subgroup sample counts**, and optional **multiple-testing corrections** (Benjamini-Hochberg).

Built for real-world engineering workflows, `fairmeasure` includes turnkey quality gates for CI/CD pipelines (`assert_fairness`), automated experiment logging for **MLflow**, and publication-ready visualizations.

---

## Why fairmeasure?

Auditing machine learning fairness in production frequently runs into practical obstacles:

1. **Fragmented Ecosystems**: Fairlearn expects scikit-learn metrics and `MetricFrame` wrappers; Aequitas demands relational SQL/crosstab style DataFrames; AIF360 relies on custom dataset classes. Switching between or combining metrics requires completely restructuring your evaluation pipeline.
2. **The Point-Estimate Fallacy**: Reporting a single disparity number (e.g., `"demographic parity difference = 0.06"`) is dangerously uninformative without confidence intervals. Is the disparity genuine, or is it statistical noise caused by small sample sizes in protected subgroups?
3. **Missing Effect Sizes**: Raw differences obscure relative risk. A 5% gap on a 5% baseline rate represents a **2.0x disparity ratio** (severe impact), whereas a 5% gap on a 50% baseline represents a **1.1x ratio**. Relative effect sizes are vital for regulatory and legal compliance.
4. **Multiple Testing Inflation**: Slicing models across 5 sensitive attributes and their intersections generates dozens of simultaneous hypothesis tests. Without False Discovery Rate (FDR) adjustments, false alarms are inevitable.
5. **Production Isolation**: Fairness audits often remain trapped inside one-off exploratory notebooks. Teams lack standard assertions for `pytest` or tracking hooks for MLOps platforms.

`fairmeasure` bridges these gaps with a single, clean API designed for data science and MLOps teams.

---

## Installation

Requires **Python 3.10+**.

```bash
# Core package (NumPy, Pandas, Scikit-Learn, Fairlearn, Aequitas, Matplotlib)
pip install fairmeasure

# With MLflow experiment tracking support
pip install "fairmeasure[mlflow]"

# For development and running the test suite
pip install "fairmeasure[dev]"
```

Or install from source:

```bash
git clone https://github.com/fairmeasure/fairmeasure.git
cd fairmeasure
pip install -e .
```

---

## Quickstart

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fairmeasure import FairnessAnalyzer

# 1. Train your model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# 2. Run the fairness audit
analyzer = FairnessAnalyzer(
    task="classification",
    n_bootstrap=1000,       # Resamples for percentile confidence intervals
    min_group_size=30,      # Automatically exclude groups with insufficient support
    random_state=42,
)

report = analyzer.analyze(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=X_test[["sex", "race"]],
    intersectional=True,    # Also audits composite groups, e.g. "sex=F & race=Black"
)

# 3. Inspect results
print(report)
```

Output:
```text
demographic_parity_difference [sex] = 0.1852 (95% CI 0.1539 to 0.2214), risk_ratio = 2.064
equalized_odds_difference [sex] = 0.2714 (95% CI 0.2095 to 0.3208), risk_ratio = 1.945
demographic_parity_difference [race] = 0.0821 (95% CI 0.0410 to 0.1245), risk_ratio = 1.280
...
```

Export to a Pandas DataFrame or JSON dictionary:

```python
df_summary = report.to_frame()
report_dict = report.to_dict()
```

---

## Supported Metrics & Backends

All difference metrics are calculated as the maximum gap between any two evaluated subgroups ($\max_g - \min_g$), meaning **0.0 represents perfect parity**.

| Metric Name | Learning Task | Default Backend (`auto`) | Underlying Group Quantity | Effect Size Ratio |
|---|---|---|---|---|
| `demographic_parity_difference` | Classification | **Fairlearn** | Positive selection rate $P(\hat{Y}=1)$ | Risk Ratio |
| `equalized_odds_difference` | Classification | **Fairlearn** | $\max(\Delta \text{TPR}, \Delta \text{FPR})$ | Risk Ratio |
| `equal_opportunity_difference` | Classification | **Fairlearn** | True Positive Rate (Recall / Sensitivity) | Risk Ratio |
| `fpr_difference` | Classification | **Aequitas** | False Positive Rate (Predictive Equality) | Risk Ratio |
| `fdr_difference` | Classification | **Aequitas** | False Discovery Rate (Predictive Parity) | Risk Ratio |
| `mae_difference` | Regression | **Fairlearn** | Mean Absolute Error (MAE) | Error Ratio |
| `exposure_parity_difference` | Ranking | **Native (NumPy)** | Logarithmic Position Exposure: $\frac{1}{\log_2(1 + r)}$ | Exposure Ratio |

### Backend Resolution

When `backend="auto"` (default), `fairmeasure` routes each metric to its preferred installed engine. You can also force a specific backend:

```python
FairnessAnalyzer(backend="fairlearn")  # Use Fairlearn exclusively
FairnessAnalyzer(backend="native")     # Pure NumPy execution (zero third-party dependencies)
```

The native NumPy backend covers every metric and acts as an automatic fallback if a third-party library is not installed in the environment.

---

## Statistical Methodology

### 1. Bootstrap Confidence Intervals
Point estimates can be misleading when evaluating small or unbalanced protected classes. `fairmeasure` uses empirical percentile bootstrap resampling (Efron & Tibshirani, 1993):
- The dataset rows are resampled with replacement $B$ times (default `n_bootstrap=500`, recommended `1000+` for reports).
- On each resample, both the metric disparity and the standardized effect size are computed.
- The 2.5th and 97.5th percentiles form the 95% confidence interval (`ci_low`, `ci_high`).
- Resamples that yield degenerate subgroup counts are excluded and tracked via `n_valid_bootstrap`. A warning is issued if >10% of resamples are unviable.

### 2. Standardized Effect Sizes
In addition to absolute difference metrics, `fairmeasure` computes relative effect sizes ($\ge 1.0$, where $1.0$ indicates parity):
- **Classification (Risk Ratio)**: $\frac{\max_g \text{Rate}_g}{\min_g \text{Rate}_g}$. For demographic parity, this is the ratio of highest to lowest selection rate. (If the lowest rate is zero, the ratio is reported as $\infty$).
- **Regression (Error Ratio)**: $\frac{\text{worst group MAE}}{\text{best group MAE}}$.
- **Ranking (Exposure Ratio)**: $\frac{\text{highest group mean exposure}}{\text{lowest group mean exposure}}$.

### 3. Intersectional Fairness & Small-Group Filtering
Real-world discrimination often compounds across multiple attributes (e.g., race *and* gender):
- Setting `intersectional=True` automatically constructs interaction groups (e.g., `"sex=F & race=Black"`).
- To prevent statistical instability from tiny subgroups, `min_group_size=30` automatically excludes groups with fewer rows than the threshold. Excluded groups and their sample sizes are recorded in `result.excluded_groups`.

### 4. Multiple Comparisons Correction (Benjamini-Hochberg)
Auditing many metrics across intersecting attributes increases family-wise error rates. Set `correction="benjamini-hochberg"` to enable sharp permutation tests and false discovery rate (FDR) control:
- Under the sharp null hypothesis, group labels are permuted across observations to construct the null disparity distribution.
- Empirical permutation p-values are calculated for every result.
- The Benjamini-Hochberg (1995) step-up procedure adjusts p-values (`p_adjusted`) across the entire report.

```python
analyzer = FairnessAnalyzer(correction="benjamini-hochberg", n_permutations=1000)
report = analyzer.analyze(y_true, y_pred, sensitive_features)
print(report[0].p_adjusted)
```

---

## Ranking and Regression

### Regression Audits
Measures disparity in continuous target predictions using Mean Absolute Error:

```python
analyzer = FairnessAnalyzer(task="regression")
report = analyzer.analyze(y_true=y_actual_salary, y_pred=y_predicted_salary, sensitive_features=df["gender"])
print(report.get("mae_difference"))
```

### Ranking Audits
Audits recommendation or search rank fairness. Scores are ordered from highest to lowest, where the item at rank $r$ receives logarithmic exposure:
$$\text{Exposure}(r) = \frac{1}{\log_2(1 + r)}$$

```python
analyzer = FairnessAnalyzer(task="ranking")
# Pass None for y_true, and ranking scores for y_pred
report = analyzer.analyze(y_true=None, y_pred=recommendation_scores, sensitive_features=df["author_group"])
print(report.get("exposure_parity_difference"))
```

---

## MLOps & CI/CD Quality Gates

### Automated Test Assertions (`assert_fairness`)
Integrate fairness regression testing into your automated test suite (`pytest`):

```python
# tests/test_model_fairness.py
from fairmeasure import FairnessAnalyzer, assert_fairness

def test_model_meets_fairness_criteria(production_model, test_dataset):
    y_pred = production_model.predict(test_dataset.X)
    report = FairnessAnalyzer(n_bootstrap=1000, random_state=42).analyze(
        y_true=test_dataset.y,
        y_pred=y_pred,
        sensitive_features=test_dataset[["gender", "ethnicity"]],
        intersectional=True,
    )

    # Fail test if demographic parity difference exceeds 0.10 or equalized odds exceeds 0.15
    assert_fairness(
        report,
        threshold={
            "demographic_parity_difference": 0.10,
            "equalized_odds_difference": 0.15,
        },
        max_effect_size=1.35,  # Also cap relative risk ratio at 1.35
        conservative=True,     # Validates upper CI bound instead of point estimate!
    )
```

If a constraint is violated, `FairnessAssertionError` (a subclass of `AssertionError`) provides an informative breakdown:

```text
FairnessAssertionError: Fairness assertion failed:
  demographic_parity_difference [gender]: CI upper bound 0.1245 > allowed threshold 0.10
  equalized_odds_difference [gender & ethnicity]: risk_ratio 1.482 > allowed 1.35
```

### MLflow Experiment Tracking (`log_to_mlflow`)
Record full fairness audits alongside your training runs:

```python
import mlflow
from fairmeasure import log_to_mlflow

with mlflow.start_run():
    # Train and evaluate model
    ...
    report = analyzer.analyze(y_true, y_pred, sensitive_features)

    # Log scalar metrics and full JSON audit report artifact
    log_to_mlflow(report, step=epoch)
```

Logged metrics include:
- `fairness.<attribute>.<metric>.value`
- `fairness.<attribute>.<metric>.ci_low`
- `fairness.<attribute>.<metric>.ci_high`
- `fairness.<attribute>.<metric>.effect_size`
- `fairness.<attribute>.<metric>.group_size.<group>`
- Artifact: `fairness_report.json`

---

## Visualizations

`fairmeasure` includes visualization helpers with publication-quality formatting:

```python
from fairmeasure import plot_report, plot_group_values, save_report_plots

# 1. Forest plot showing metric point estimates, CIs, effect sizes, and threshold line
fig = plot_report(report, threshold=0.10, save_path="audit_summary.png")

# 2. Subgroup breakdown bar chart with sample sizes (n)
fig = plot_group_values(report.get("demographic_parity_difference", "sex"), save_path="sex_parity.png")

# 3. Export all plots into an audit directory
saved_files = save_report_plots(report, out_dir="audit_reports/", threshold=0.10)
```

---

## Performance & Parallelization

Bootstrap resampling invokes metric evaluations $B$ times. With multiple metrics, attributes, and 1000 resamples, sequential execution can take several seconds.

Use `n_jobs=-1` to parallelize resampling across all CPU cores:

```python
analyzer = FairnessAnalyzer(n_bootstrap=1000, n_jobs=-1)
```

> **Note on Windows pytest runs**: When executing multi-process tests with Joblib inside `pytest` on Windows, invoke pytest with output capturing disabled (`pytest -s`) to prevent process handle collisions.

---

## Project Structure

```
fairmeasure/
├── analyzer.py        # Core FairnessAnalyzer orchestration engine
├── backends.py        # Fairlearn, Aequitas, and Native NumPy execution adapters
├── metrics.py         # Metric specifications, registry, and vectorized group rates
├── results.py         # Structured containers: MetricResult, EffectSize, AuditReport
├── stats.py           # Percentile bootstrap, permutation tests, Benjamini-Hochberg FDR
├── integrations.py    # CI/CD assertions (assert_fairness) and MLflow tracking (log_to_mlflow)
└── viz.py             # Forest plots and group distribution charts
tests/                 # Pytest test suite (unit tests, statistical coverage, integrations)
demo.ipynb             # End-to-end interactive tutorial notebook
```

---

## References & Academic Grounding

- **Bootstrap Confidence Intervals**: Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
- **FDR Multiple Testing Correction**: Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
- **Fairness Criteria**: Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems (NeurIPS)*.
- **Fairlearn**: Bird, S. et al. (2020). *Fairlearn: A toolkit for assessing and improving fairness in AI*. Microsoft Technical Report.
- **Aequitas**: Saleiro, P. et al. (2018). *Aequitas: A Bias and Fairness Audit Toolkit*. arXiv:1811.05577.

---

## License

This project is licensed under the [MIT License](LICENSE).
