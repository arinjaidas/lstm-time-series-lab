# Data dictionary

| Field or concept | Definition | Unit or rule |
| --- | --- | --- |
| Date | Unique increasing observation date | At least 180 observations |
| log_return | Log return for the period ending on Date | Finite scalar |
| lookback | Historical observations per sequence | 12 in the public example |
| forecast target | Next log return | Target excluded from input window |
| cost_bps | Cost per unit turnover | 5 basis points in the example |

New NumPy implementation with synthetic returns, verified gradients, chronological splits and benchmark reporting.

The default example is synthetic or illustrative. Real-data inputs require their own permissions, definitions and source audit. Missing or infinite values are not evidence of a valid measured zero.
