"""A small NumPy LSTM with chronological evaluation and a synthetic trading example.

New 2026 public reconstruction inspired by the documented 2023 prototype.
No original model weights, market data or historical performance are claimed.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def windows(series, lookback=12):
    values = np.asarray(series, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not isinstance(lookback, int) or lookback < 1 or len(values) <= lookback:
        raise ValueError('Finite scalar series and a shorter positive lookback required.')
    target_index = np.arange(lookback, len(values))
    x = np.array([values[t - lookback:t] for t in target_index])[:, :, None]
    return x, values[target_index], target_index


def scaler(series, train_end):
    values = np.asarray(series, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not isinstance(train_end, int) or not 3 <= train_end <= len(values):
        raise ValueError('Training cutoff must select at least three finite observations.')
    mean, sd = values[:train_end].mean(), values[:train_end].std(ddof=1)
    if sd < 1e-12: raise ValueError('Training series must vary.')
    return float(mean), float(sd)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


class LSTM:
    def __init__(self, hidden=8, seed=20261019):
        if not isinstance(hidden, int) or hidden < 1: raise ValueError('Hidden size must be a positive integer.')
        self.hidden = hidden
        rng = np.random.default_rng(seed)
        self.params = {'W': rng.normal(0, .15, (hidden + 1, 4 * hidden)),
                       'b': np.zeros(4 * hidden), 'V': rng.normal(0, .15, hidden), 'bo': np.zeros(1)}
        self.params['b'][hidden:2 * hidden] = 1

    def forward(self, x, cache=False):
        x = np.asarray(x, dtype=float)
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != 1 or not np.isfinite(x).all():
            raise ValueError('Expected a finite nonempty (batch, time, 1) array.')
        size = self.hidden; h = np.zeros((len(x), size)); c = h.copy(); saved = []
        for t in range(x.shape[1]):
            joined = np.concatenate([x[:, t], h], axis=1)
            gates = joined @ self.params['W'] + self.params['b']
            i = sigmoid(gates[:, :size]); f = sigmoid(gates[:, size:2 * size])
            g = np.tanh(gates[:, 2 * size:3 * size]); o = sigmoid(gates[:, 3 * size:])
            previous_c = c
            c = f * previous_c + i * g
            h = o * np.tanh(c)
            if cache: saved.append((joined, i, f, g, o, previous_c, c))
        prediction = h @ self.params['V'] + self.params['bo'][0]
        return (prediction, h, saved) if cache else prediction

    def loss_gradient(self, x, y):
        prediction, final_h, saved = self.forward(x, cache=True)
        y = np.asarray(y, dtype=float)
        if y.shape != prediction.shape or not np.isfinite(y).all(): raise ValueError('Finite aligned targets required.')
        difference = prediction - y
        derivative = 2 * difference / len(y)
        gradients = {key: np.zeros_like(value) for key, value in self.params.items()}
        gradients['V'] = final_h.T @ derivative; gradients['bo'][0] = derivative.sum()
        dh = derivative[:, None] * self.params['V'][None, :]
        dc_next = np.zeros_like(dh)
        for joined, i, f, g, o, previous_c, c in reversed(saved):
            tanh_c = np.tanh(c)
            dc = dc_next + dh * o * (1 - tanh_c ** 2)
            gates_derivative = np.concatenate([
                dc * g * i * (1 - i),
                dc * previous_c * f * (1 - f),
                dc * i * (1 - g ** 2),
                dh * tanh_c * o * (1 - o),
            ], axis=1)
            gradients['W'] += joined.T @ gates_derivative
            gradients['b'] += gates_derivative.sum(axis=0)
            dh = (gates_derivative @ self.params['W'].T)[:, 1:]
            dc_next = dc * f
        return float(np.mean(difference ** 2)), gradients


def train(model, x, y, validation_x, validation_y, epochs=100, batch_size=64, learning_rate=.01, patience=15, seed=20261019):
    if epochs < 1 or batch_size < 1 or patience < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Positive training settings required.')
    rng = np.random.default_rng(seed)
    first = {k: np.zeros_like(v) for k, v in model.params.items()}
    second = {k: np.zeros_like(v) for k, v in model.params.items()}
    best_loss = np.inf; best_params = None; history = []; step = 0; stale = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            indices = order[start:start + batch_size]
            _, gradients = model.loss_gradient(x[indices], y[indices])
            norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients.values()))
            if not np.isfinite(norm): raise FloatingPointError('Nonfinite training gradient.')
            step += 1
            for key in model.params:
                g = gradients[key] / max(1, norm)
                first[key] = .9 * first[key] + .1 * g
                second[key] = .999 * second[key] + .001 * g ** 2
                model.params[key] -= learning_rate * (first[key] / (1 - .9 ** step)) / (np.sqrt(second[key] / (1 - .999 ** step)) + 1e-8)
        loss = float(np.mean((model.forward(x) - y) ** 2))
        validation_loss = float(np.mean((model.forward(validation_x) - validation_y) ** 2))
        history.append({'epoch': epoch, 'training_mse': loss, 'validation_mse': validation_loss})
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss; best_params = {k: v.copy() for k, v in model.params.items()}; stale = 0
        else:
            stale += 1
        if stale >= patience: break
    model.params = best_params
    return pd.DataFrame(history)


def long_cash(actual_log_returns, predicted_log_returns, cost_bps=5):
    actual = np.asarray(actual_log_returns, dtype=float)
    predicted = np.asarray(predicted_log_returns, dtype=float)
    if actual.ndim != 1 or not len(actual) or actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Finite aligned return and forecast arrays required.')
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError('Transaction cost must lie in [0, 10000) basis points.')
    simple = np.expm1(actual)
    if not np.isfinite(simple).all(): raise ValueError('Returns exceed numerical range.')
    position = (predicted > 0).astype(float)
    turnover = np.abs(np.diff(np.r_[0., position]))
    net = (1 - cost_bps / 10000 * turnover) * (1 + position * simple) - 1
    buy_hold = simple.copy(); buy_hold[0] = (1 - cost_bps / 10000) * (1 + simple[0]) - 1
    return pd.DataFrame({'actual_log_return': actual, 'forecast_log_return': predicted, 'position': position,
                         'turnover': turnover, 'net_simple_return': net, 'strategy_wealth': np.cumprod(1 + net),
                         'buy_hold_wealth': np.cumprod(1 + buy_hold)})


def synthetic_returns(seed=20261019, n=720):
    rng = np.random.default_rng(seed); values = np.zeros(n)
    for t in range(1, n): values[t] = .00015 + .15 * values[t - 1] + .001 * np.sin(t / 10) + rng.normal(0, .01)
    return pd.Series(values, index=pd.bdate_range('2020-01-01', periods=n), name='log_return')


def run(series, output, synthetic=False, epochs=100):
    if len(series) < 180 or series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValueError('At least 180 uniquely dated ordered observations required.')
    values = series.to_numpy(dtype=float); n = len(values)
    train_end, validation_end = int(.6 * n), int(.8 * n)
    mean, sd = scaler(values, train_end)
    x, y, target = windows((values - mean) / sd, 12)
    train_mask, val_mask, test_mask = target < train_end, (target >= train_end) & (target < validation_end), target >= validation_end
    model = LSTM()
    losses = train(model, x[train_mask], y[train_mask], x[val_mask], y[val_mask], epochs=epochs)
    prediction = model.forward(x[test_mask]) * sd + mean
    actual = values[target[test_mask]]
    ar_x = np.column_stack([np.ones(train_end - 1), values[:train_end - 1]])
    ar_beta = np.linalg.lstsq(ar_x, values[1:train_end], rcond=None)[0]
    ar_prediction = ar_beta[0] + ar_beta[1] * values[target[test_mask] - 1]
    predictions = {'lstm': prediction, 'zero_return': np.zeros(len(actual)), 'ar1': ar_prediction}
    scores = []
    for method, pred in predictions.items():
        error = actual - pred
        scores.append({'method': method, 'test_rows': len(actual), 'mae': np.mean(np.abs(error)), 'rmse': np.sqrt(np.mean(error ** 2))})
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    losses.to_csv(out / 'training_history.csv', index=False)
    pd.DataFrame(scores).to_csv(out / 'test_metrics.csv', index=False)
    forecasts = pd.DataFrame({'actual_log_return': actual, **predictions}, index=series.index[target[test_mask]])
    forecasts.to_csv(out / 'test_forecasts.csv', index_label='Date')
    strategy = long_cash(actual, prediction)
    strategy.index = forecasts.index; strategy.to_csv(out / 'synthetic_strategy.csv' if synthetic else out / 'strategy.csv', index_label='Date')
    if synthetic: series.to_csv(out / 'synthetic_returns.csv', index_label='Date')
    state = {'hidden': model.hidden, 'scaler_mean': mean, 'scaler_sd': sd, 'lookback': 12,
             'params': {k: v.tolist() for k, v in model.params.items()}}
    (out / 'model_state.json').write_text(json.dumps(state, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    losses.plot(x='epoch', y=['training_mse', 'validation_mse'], ax=axes[0])
    strategy[['strategy_wealth', 'buy_hold_wealth']].plot(ax=axes[1])
    axes[0].set(title='Validation-based early stopping', ylabel='Standardized-return MSE')
    axes[1].set(title='Synthetic long/cash illustration; 5 bps cost', ylabel='Wealth multiple')
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('SYNTHETIC NUMPY LSTM DEMONSTRATION'); fig.tight_layout()
    fig.savefig(out / 'lstm_evaluation.svg'); plt.close(fig)
    meta = {'synthetic': synthetic, 'seed': 20261019, 'rows': n, 'train_rows': train_end,
            'validation_rows': validation_end - train_end, 'test_rows': n - validation_end,
            'train_end': str(series.index[train_end - 1].date()), 'test_start': str(series.index[validation_end].date()),
            'hidden_units': 8, 'lookback': 12, 'epochs_run': len(losses), 'selected_epoch': int(losses.loc[losses.validation_mse.idxmin(), 'epoch']),
            'scaling': 'Fit using training observations only.', 'selection': 'Early stopping on validation MSE; test targets unused during fitting.',
            'backtest': 'Long/cash; signal uses past returns; 5 bps per unit turnover; no final liquidation, spread, cash interest or market impact.',
            'scope': 'New public reconstruction; not the original 2023 local model.',
            'interpretation': 'Synthetic prediction and trading workflow, not evidence of market alpha or historical investment performance.'}
    (out / 'run_metadata.json').write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--demo', action='store_true'); g.add_argument('--data', type=Path)
    p.add_argument('--output', default='demo_output'); p.add_argument('--epochs', type=int, default=100)
    a = p.parse_args()
    series = synthetic_returns() if a.demo else pd.read_csv(a.data, index_col='Date', parse_dates=True)['log_return']
    print(json.dumps(run(series, a.output, a.demo, a.epochs), indent=2))
