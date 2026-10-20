![LSTM Time-Series and Trading Workflow](assets/header.svg)

# LSTM Time-Series and Trading Workflow

A NumPy LSTM with checked backpropagation, chronological validation, simple forecasting baselines and a synthetic cost-aware trading illustration.

**Research question:** How can a sequence model be trained and evaluated without letting future observations leak into features, scaling or model selection?

**Public edition:** New NumPy implementation with synthetic returns, verified gradients, chronological splits and benchmark reporting.

**8 functional checks passed locally**, plus a complete demonstration run.

## What you can inspect

- Implement an input, forget, candidate and output gate with backpropagation through time.
- Check analytical gradients against independent finite differences.
- Fit scaling to the first 60 percent of observations, validate on the next 20 percent and test on the final 20 percent.
- Use validation-based early stopping and keep test targets out of fitting.
- Compare forecasts with zero-return and AR1 baselines, then illustrate long/cash turnover costs separately.

## Results preview

All values below are synthetic or illustrative workflow results.

![LSTM Time-Series and Trading Workflow results preview](demo_output/lstm_evaluation.svg)

| Method | Test observations | MAE of log return | RMSE of log return |
| --- | --- | --- | --- |
| lstm | 144 | 0.00784278 | 0.00959432 |
| zero_return | 144 | 0.00796721 | 0.00968691 |
| ar1 | 144 | 0.00784283 | 0.00960015 |

The model used 432 training observations, 144 validation observations and 144 test observations. The saved epoch is 5. Small differences among methods are not evidence of robust market alpha. The trading chart is a separate synthetic illustration.

## Run the example

Use Python 3.12. From this project's directory:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -v
python lstm_lab.py --demo --output demo_output
```

For an isolated environment, run `python -m venv .venv` first. Activate it with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\activate` on Windows. The standard example runs offline after dependencies are installed.

The [notebook walkthrough](notebooks/01_results_walkthrough.ipynb) explains the saved outputs. Exact command-line options appear in [CLI.txt](docs/CLI.txt).

## Documentation

| Guide | What it covers |
| --- | --- |
| [Methods](docs/METHODS.md) | Formulas, assumptions and interpretation |
| [Data dictionary](docs/DATA_DICTIONARY.md) | Variables, units and validation rules |
| [Results](docs/RESULTS.md) | Actual bundled output and how to read it |
| [Reproducibility](REPRODUCIBILITY.md) | Commands, checks and the recorded environment |
| [Provenance and attribution](docs/PROVENANCE.md) | Historical context, reconstruction and contribution |
| [Roadmap](docs/ROADMAP.md) | Specific extensions, with unfinished work labelled |
| [References](docs/REFERENCES.md) | Method documentation and source links |

## Limits of interpretation

- The series is synthetic and the test sample is small.
- A small RMSE difference is not evidence of statistical or economic superiority.
- The trading illustration omits final liquidation, spread, market impact and cash interest.
- The model is a compact educational implementation, not a production trading system.

## Contribution and research context

Documented October 2023 LSTM prediction and trading prototype. The historical model's execution and original local files were not recovered.

The earlier ChatGPT work contained prediction and trading code. The October 2026 public reconstruction makes the model and evaluation inspectable without claiming recovered weights, prior successful trading or historical investment results.

**Skills demonstrated:** Python, NumPy, LSTM, backpropagation, machine learning, temporal validation, backtesting, transaction costs.

[Portfolio](https://github.com/arinjaidas/My-Projects) · [GitHub profile](https://github.com/arinjaidas)
