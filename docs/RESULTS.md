# Results walkthrough

All values below are synthetic or illustrative workflow results.

| Method | Test observations | MAE of log return | RMSE of log return |
| --- | --- | --- | --- |
| lstm | 144 | 0.00784278 | 0.00959432 |
| zero_return | 144 | 0.00796721 | 0.00968691 |
| ar1 | 144 | 0.00784283 | 0.00960015 |

The model used 432 training observations, 144 validation observations and 144 test observations. The saved epoch is 5. Small differences among methods are not evidence of robust market alpha. The trading chart is a separate synthetic illustration.

![Results preview](../demo_output/lstm_evaluation.svg)

The numerical files in [demo_output](../demo_output/) are the authoritative bundled artifacts. Re-run the entry command to reproduce the workflow. Figure metadata can differ across renders even when the underlying numerical example is unchanged.
