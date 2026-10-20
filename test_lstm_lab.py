import unittest
import numpy as np
from lstm_lab import LSTM, windows, scaler, long_cash, train


class LSTMTests(unittest.TestCase):
    def test_gradient_against_finite_difference(self):
        model = LSTM(hidden=2, seed=1)
        x = np.array([[[.1], [.2], [-.2]], [[.4], [-.1], [.3]]]); y = np.array([.2, -.1])
        _, gradients = model.loss_gradient(x, y)
        epsilon = 1e-6
        for key, values in model.params.items():
            for index in list(np.ndindex(values.shape))[::max(1, values.size // 8)]:
                old = values[index]
                values[index] = old + epsilon; plus = np.mean((model.forward(x) - y) ** 2)
                values[index] = old - epsilon; minus = np.mean((model.forward(x) - y) ** 2)
                values[index] = old
                self.assertAlmostEqual((plus - minus) / (2 * epsilon), gradients[key][index], delta=2e-7)

    def test_window_excludes_target(self):
        x, y, indices = windows(np.arange(20.), 4)
        for row, t in enumerate(indices):
            self.assertEqual(x[row, -1, 0], t - 1)
            self.assertEqual(y[row], t)

    def test_future_cannot_affect_scaler(self):
        a = np.arange(100.); b = a.copy(); b[60:] = 10000
        self.assertEqual(scaler(a, 60), scaler(b, 60))

    def test_batch_predictions_are_independent(self):
        model = LSTM(hidden=2); x = np.arange(15.).reshape(3, 5, 1) / 10
        np.testing.assert_allclose(model.forward(x), [model.forward(x[i:i + 1])[0] for i in range(3)], atol=1e-14)

    def test_transaction_cost_and_turnover(self):
        result = long_cash(np.zeros(4), [1, 1, -1, 1], cost_bps=10)
        np.testing.assert_array_equal(result.turnover, [1, 0, 1, 1])
        self.assertAlmostEqual(result.strategy_wealth.iloc[-1], .999 ** 3)

    def test_always_long_matches_buy_hold(self):
        out = long_cash([.01, -.02, .03], [1, 1, 1])
        np.testing.assert_allclose(out.strategy_wealth, out.buy_hold_wealth)

    def test_training_learns_simple_target(self):
        x = np.zeros((16, 3, 1)); y = np.full(16, .5); model = LSTM(hidden=2)
        initial = np.mean((model.forward(x) - y) ** 2)
        train(model, x, y, x, y, epochs=40, patience=40)
        self.assertLess(np.mean((model.forward(x) - y) ** 2), initial / 10)

    def test_invalid_shapes_rejected(self):
        with self.assertRaises(ValueError): LSTM().forward(np.ones((4, 5)))
        with self.assertRaises(ValueError): long_cash([.1], [.2], cost_bps=-1)


if __name__ == '__main__': unittest.main()
