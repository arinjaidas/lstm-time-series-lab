# Methods

## Sequence construction and gates

A target at observation t receives only observations t-12 through t-1 as input. Each sequence starts with zero hidden and cell states. Input, forget and output gates use sigmoid activations, while the candidate uses tanh. The cell updates as `c_t=f_t*c_(t-1)+i_t*g_t`; hidden state is `h_t=o_t*tanh(c_t)`. A linear output predicts the standardised next log return.

Backpropagation passes through all sequence steps. The functional checks compare selected parameter derivatives with independent central finite differences. Adam uses first- and second-moment updates and a global gradient-norm cap of one. This compact implementation is intended to make the learning calculation inspectable.

## Temporal evaluation

The first 60 percent of observations fit the mean and standard deviation and supply training targets. The next 20 percent supply validation targets. Validation MSE selects the saved model through early stopping, with patience 15. The final 20 percent are reserved for testing. Past realised validation or test returns can appear in later input windows because they would be known then; their targets do not enter model fitting.

The zero-return baseline and training-fitted AR1 are reported alongside LSTM test MAE/RMSE. A single seeded test split does not establish reliable superiority. The synthetic generator includes mild autoregression and noise and was not chosen as a claim about actual market predictability.

## Trading illustration

A positive forecast takes a fully invested long position; otherwise capital remains in cash. Turnover is the absolute change from the prior position, starting flat. Net growth multiplies `(1-fee*turnover)` by `(1+position*realised_simple_return)`. Buy-and-hold pays the same initial entry fee. Neither strategy is liquidated at the final observation, and cash interest, spread and market impact are omitted.
