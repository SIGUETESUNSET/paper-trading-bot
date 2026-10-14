import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from paper_trading import (
    Bar,
    DataValidationError,
    _data_age_days,
    backtest,
    load_bars,
    rank_parameters,
    screen_parameters,
)
from market_data import MarketDataDownloadError, parse_chart_response


def bars_from_closes(closes, opens=None):
    opens = opens or closes
    return [
        Bar(
            timestamp=f"2024-01-{index + 1:02d}",
            open=float(open_price),
            high=float(max(open_price, close)),
            low=float(min(open_price, close)),
            close=float(close),
            volume=100.0,
        )
        for index, (open_price, close) in enumerate(zip(opens, closes))
    ]


class BacktestTests(unittest.TestCase):
    def test_crossover_executes_at_next_bar_open(self):
        data = bars_from_closes([3, 2, 1, 2, 4], [3, 2, 1, 2, 10])
        result = backtest(
            data, starting_cash=1000, allocation=1, max_allocation=1,
            fee_rate=0, slippage_rate=0, drawdown_limit=None,
            fast_window=1, slow_window=2
        )
        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(trade.status, "open")
        self.assertEqual(trade.entry_time, "2024-01-05")
        self.assertEqual(trade.entry_price, 10.0)
        self.assertEqual(result.ending_equity, 400.0)
        self.assertEqual(result.trade_count, 0)

    def test_exit_and_buy_sell_fees_are_accounted_for(self):
        data = bars_from_closes([3, 2, 1, 2, 4, 2, 1], [3, 2, 1, 2, 4, 4, 2])
        result = backtest(
            data,
            starting_cash=1000,
            allocation=0.5,
            max_allocation=0.5,
            fee_rate=0.01,
            slippage_rate=0,
            drawdown_limit=None,
            fast_window=1,
            slow_window=2,
        )
        self.assertEqual(result.trade_count, 1)
        trade = result.trades[0]
        self.assertEqual(trade.status, "closed")
        self.assertEqual(trade.entry_time, "2024-01-05")
        self.assertEqual(trade.exit_time, "2024-01-07")
        self.assertAlmostEqual(trade.quantity, (500 / 1.01) / 4)
        self.assertAlmostEqual(trade.entry_fee, trade.quantity * 4 * 0.01)
        self.assertAlmostEqual(trade.exit_fee, trade.quantity * 2 * 0.01)
        self.assertAlmostEqual(
            trade.net_pnl, trade.quantity * (2 - 4) - trade.entry_fee - trade.exit_fee
        )

    def test_metrics_include_initial_equity_and_mark_to_close(self):
        data = bars_from_closes([3, 2, 1, 2, 4], [3, 2, 1, 2, 10])
        result = backtest(
            data, starting_cash=1000, allocation=1, max_allocation=1,
            fee_rate=0, slippage_rate=0, drawdown_limit=None,
            fast_window=1, slow_window=2
        )
        self.assertAlmostEqual(result.total_return, -0.6)
        self.assertAlmostEqual(result.max_drawdown, 0.6)

    def test_allocation_changes_position_size(self):
        data = bars_from_closes([3, 2, 1, 2, 4], [3, 2, 1, 2, 10])
        result = backtest(
            data, starting_cash=1000, allocation=0.25, fee_rate=0,
            max_allocation=0.25, slippage_rate=0, drawdown_limit=None,
            fast_window=1, slow_window=2
        )
        self.assertEqual(result.trades[0].quantity, 25.0)
        self.assertEqual(result.ending_equity, 850.0)

    def test_rejects_out_of_order_csv(self):
        content = (
            "timestamp,open,high,low,close,volume\n"
            "2024-01-02,1,2,1,2,10\n"
            "2024-01-01,2,3,1,2,10\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(DataValidationError, "strictly chronological"):
                load_bars(path)

    def test_rejects_invalid_ohlc_and_missing_columns(self):
        examples = [
            ("timestamp,open,high,low,close,volume\n2024-01-01,3,2,1,3,10\n", "inconsistent OHLC"),
            ("timestamp,open,high,low,close\n2024-01-01,1,2,1,2\n", "missing required columns"),
            ("timestamp,open,high,low,close,volume\n2024-01-01,NaN,2,1,2,10\n", "finite"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, (content, message) in enumerate(examples):
                path = Path(directory) / f"bad-{index}.csv"
                path.write_text(content, encoding="utf-8")
                with self.subTest(message=message):
                    with self.assertRaisesRegex(DataValidationError, message):
                        load_bars(path)

    def test_parameter_validation(self):
        data = bars_from_closes([10])
        with self.assertRaisesRegex(ValueError, "fast window"):
            backtest(data, fast_window=3, slow_window=2)
        with self.assertRaisesRegex(ValueError, "allocation"):
            backtest(data, allocation=0)
        with self.assertRaisesRegex(ValueError, "start index"):
            backtest(data, start_index=1)

    def test_screening_uses_separate_validation_and_200_dollar_capital(self):
        closes = [10, 9, 8, 9, 11, 13, 12, 10, 8, 9, 12, 14, 13, 10, 8, 9, 12, 15, 14, 11]
        data = bars_from_closes(closes)
        arguments = dict(
            starting_cash=200,
            allocation=1,
            max_allocation=1,
            fee_rate=0.001,
            slippage_rate=0,
            drawdown_limit=None,
            max_window=5,
        )
        fast, slow, training, validation, _, baseline_validation = screen_parameters(
            data, **arguments
        )
        split = int(len(data) * 0.7)
        self.assertLess(fast, slow)
        self.assertEqual(training.starting_cash, 200)
        self.assertEqual(validation.starting_cash, 200)
        self.assertEqual(baseline_validation.starting_cash, 200)
        self.assertTrue(
            all(trade.entry_time >= data[split].timestamp for trade in validation.trades)
        )

        changed_validation = data[:split] + [
            Bar(
                timestamp=bar.timestamp,
                open=bar.open * 3,
                high=bar.high * 3,
                low=bar.low * 3,
                close=bar.close * 3,
                volume=bar.volume,
            )
            for bar in data[split:]
        ]
        changed = screen_parameters(changed_validation, **arguments)
        self.assertEqual((fast, slow), changed[:2])
        self.assertEqual(training.total_return, changed[2].total_return)

    def test_optimizer_ranks_net_profit_and_prefers_drawdown_feasible_candidates(self):
        closes = [
            10, 9, 8, 9, 11, 13, 12, 10, 8, 9, 12, 14, 13, 10, 8,
            9, 12, 15, 14, 11, 9, 10, 13, 16, 14, 12, 10, 11, 14, 17,
        ]
        candidates = rank_parameters(
            bars_from_closes(closes),
            starting_cash=200,
            allocation=0.25,
            max_allocation=0.25,
            fee_rate=0.001,
            slippage_rate=0.0005,
            drawdown_limit=0.15,
            max_window=6,
        )
        self.assertTrue(candidates)
        self.assertEqual(
            candidates,
            sorted(
                candidates,
                key=lambda candidate: (
                    candidate.mean_fold_return,
                    -candidate.worst_fold_drawdown,
                    candidate.training_result.total_return,
                    -candidate.fast_window,
                    -candidate.slow_window,
                ),
                reverse=True,
            ),
        )
        if any(result.worst_fold_drawdown <= 0.15 for result in candidates):
            self.assertTrue(
                all(result.worst_fold_drawdown <= 0.15 for result in candidates)
            )

    def test_position_cap_limits_exposure_and_stress_gap_loss(self):
        normal = bars_from_closes(
            [3, 2, 1, 2, 4, 4, 2, 1],
            [3, 2, 1, 2, 4, 4, 2, 2],
        )
        capped = backtest(
            normal,
            starting_cash=200,
            allocation=1,
            max_allocation=0.25,
            fee_rate=0,
            slippage_rate=0,
            drawdown_limit=None,
            fast_window=1,
            slow_window=2,
        )
        self.assertEqual(capped.trades[0].quantity, 12.5)
        self.assertEqual(capped.ending_equity, 175)
        self.assertAlmostEqual(capped.max_drawdown, 0.125)

        shocked = normal[:5] + [
            Bar(
                timestamp=bar.timestamp,
                open=bar.open * 0.5,
                high=bar.high * 0.5,
                low=bar.low * 0.5,
                close=bar.close * 0.5,
                volume=bar.volume,
            )
            for bar in normal[5:]
        ]
        stress_result = backtest(
            shocked,
            starting_cash=200,
            allocation=1,
            max_allocation=0.25,
            fee_rate=0,
            slippage_rate=0,
            drawdown_limit=None,
            fast_window=1,
            slow_window=2,
        )
        self.assertEqual(stress_result.trades[0].net_pnl, -37.5)
        self.assertEqual(stress_result.ending_equity, 162.5)
        self.assertLess(stress_result.ending_equity, capped.ending_equity)

    def test_drawdown_circuit_breaker_exits_next_open_and_blocks_reentry(self):
        base_data = bars_from_closes(
            [3, 2, 1, 2, 4, 1, 1, 3, 4],
            [3, 2, 1, 2, 4, 1, 1, 3, 4],
        )
        data = base_data[:6] + [
            Bar(
                timestamp=base_data[6].timestamp,
                open=0.5,
                high=1,
                low=0.5,
                close=1,
                volume=base_data[6].volume,
            )
        ] + base_data[7:]
        result = backtest(
            data,
            starting_cash=200,
            allocation=1,
            max_allocation=0.25,
            fee_rate=0,
            slippage_rate=0,
            drawdown_limit=0.15,
            fast_window=1,
            slow_window=2,
        )
        self.assertTrue(result.risk_limit_triggered)
        self.assertEqual(result.max_drawdown, 0.21875)
        self.assertEqual(result.trade_count, 1)
        self.assertEqual(result.trades[0].exit_time, "2024-01-07")
        self.assertEqual(result.ending_equity, 156.25)
        self.assertEqual(len(result.trades), 1)

    def test_slippage_is_charged_on_both_sides(self):
        data = bars_from_closes(
            [3, 2, 1, 2, 4, 4, 2, 1],
            [3, 2, 1, 2, 4, 4, 2, 2],
        )
        no_slippage = backtest(
            data, starting_cash=200, allocation=1, max_allocation=0.25,
            fee_rate=0, slippage_rate=0, drawdown_limit=None,
            fast_window=1, slow_window=2,
        )
        with_slippage = backtest(
            data, starting_cash=200, allocation=1, max_allocation=0.25,
            fee_rate=0, slippage_rate=0.01, drawdown_limit=None,
            fast_window=1, slow_window=2,
        )
        self.assertLess(with_slippage.ending_equity, no_slippage.ending_equity)
        trade = with_slippage.trades[0]
        self.assertAlmostEqual(trade.entry_price, 4 * 1.01)
        self.assertAlmostEqual(trade.exit_price, 2 * 0.99)

    def test_data_age_uses_current_clock_without_network(self):
        now = datetime(2026, 10, 13, tzinfo=timezone.utc)
        self.assertAlmostEqual(
            _data_age_days("2026-10-11T00:00:00Z", now), 2.0
        )

    def test_market_feed_parser_drops_incomplete_candle_and_checks_currency(self):
        def epoch(day):
            return int(datetime(2026, 10, day, tzinfo=timezone.utc).timestamp())

        payload = {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "symbol": "BTC-USD",
                            "currency": "USD",
                            "exchangeTimezoneName": "UTC",
                        },
                        "timestamp": [epoch(11), epoch(12), epoch(13)],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [90.0, 100.0, 110.0],
                                    "high": [95.0, 105.0, 115.0],
                                    "low": [88.0, 95.0, 108.0],
                                    "close": [92.0, 103.0, 112.0],
                                    "volume": [400.0, 500.0, 200.0],
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        }
        bars = parse_chart_response(
            payload, symbol="BTC-USD", today=datetime(2026, 10, 13).date()
        )
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[-1].close, 103.0)
        payload["chart"]["result"][0]["meta"]["currency"] = "EUR"
        with self.assertRaisesRegex(MarketDataDownloadError, "quoted in EUR"):
            parse_chart_response(
                payload, symbol="BTC-USD", today=datetime(2026, 10, 13).date()
            )

    def test_market_feed_parser_rejects_invalid_ohlc(self):
        epoch = int(datetime(2026, 10, 10, tzinfo=timezone.utc).timestamp())
        payload = {
            "chart": {
                "result": [
                    {
                        "meta": {"currency": "USD", "exchangeTimezoneName": "UTC"},
                        "timestamp": [epoch, epoch + 86400],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [10, 10],
                                    "high": [9, 12],
                                    "low": [8, 9],
                                    "close": [9, 11],
                                    "volume": [1, 1],
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        }
        with self.assertRaisesRegex(MarketDataDownloadError, "failed OHLCV validation"):
            parse_chart_response(
                payload, symbol="BTC-USD", today=datetime(2026, 10, 13).date()
            )


if __name__ == "__main__":
    unittest.main()
