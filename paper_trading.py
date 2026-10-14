"""Educational long-only moving-average crossover simulator.

Signals are calculated from completed closes and executed at the following
bar's open. Market-data retrieval is optional; no broker is connected.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Sequence


class DataValidationError(ValueError):
    """Raised when a market-data CSV is malformed or inconsistent."""


@dataclass(frozen=True)
class Bar:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Trade:
    status: str
    entry_time: str
    entry_price: float
    exit_time: str | None
    exit_price: float | None
    quantity: float
    entry_fee: float
    exit_fee: float
    net_pnl: float


@dataclass(frozen=True)
class BacktestResult:
    starting_cash: float
    ending_equity: float
    total_return: float
    max_drawdown: float
    trade_count: int
    trades: tuple[Trade, ...]
    risk_limit_triggered: bool


@dataclass(frozen=True)
class ParameterScore:
    fast_window: int
    slow_window: int
    training_result: BacktestResult
    fold_results: tuple[BacktestResult, ...]

    @property
    def mean_fold_return(self) -> float:
        return sum(result.total_return for result in self.fold_results) / len(
            self.fold_results
        )

    @property
    def worst_fold_drawdown(self) -> float:
        return max(result.max_drawdown for result in self.fold_results)


def load_bars(path: str | Path) -> list[Bar]:
    """Read and validate ISO-timestamped OHLCV rows from a CSV file."""
    try:
        handle = open(path, "r", newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise DataValidationError(f"cannot read CSV file {path!s}: {exc}") from exc

    with handle:
        try:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DataValidationError("CSV is empty or missing a header row")
            headers = [name.strip().lower() if name else "" for name in reader.fieldnames]
            required = ("timestamp", "open", "high", "low", "close", "volume")
            if len(headers) != len(set(headers)):
                raise DataValidationError("CSV contains duplicate column names")
            missing = [name for name in required if name not in headers]
            if missing:
                raise DataValidationError(
                    "CSV is missing required columns: " + ", ".join(missing)
                )

            bars: list[Bar] = []
            previous_time: datetime | None = None
            previous_awareness: bool | None = None
            for row_number, raw_row in enumerate(reader, start=2):
                if None in raw_row:
                    raise DataValidationError(f"row {row_number}: too many CSV fields")
                row = {
                    (key.strip().lower() if key else ""): value
                    for key, value in raw_row.items()
                }
                if any(row.get(name) is None or not row[name].strip() for name in required):
                    raise DataValidationError(f"row {row_number}: missing required value")

                timestamp = row["timestamp"].strip()
                try:
                    parsed_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise DataValidationError(
                        f"row {row_number}: timestamp must be ISO 8601"
                    ) from exc
                is_aware = parsed_time.utcoffset() is not None
                if previous_awareness is not None and is_aware != previous_awareness:
                    raise DataValidationError(
                        f"row {row_number}: timestamps cannot mix timezones "
                        "and timezone-naive values"
                    )
                if previous_time is not None and parsed_time <= previous_time:
                    raise DataValidationError(
                        f"row {row_number}: timestamps must be strictly chronological"
                    )

                values: dict[str, float] = {}
                for name in ("open", "high", "low", "close", "volume"):
                    try:
                        value = float(row[name])
                    except (TypeError, ValueError) as exc:
                        raise DataValidationError(
                            f"row {row_number}: {name} must be a number"
                        ) from exc
                    if not math.isfinite(value):
                        raise DataValidationError(
                            f"row {row_number}: {name} must be finite"
                        )
                    values[name] = value

                if any(values[name] <= 0 for name in ("open", "high", "low", "close")):
                    raise DataValidationError(
                        f"row {row_number}: OHLC prices must be greater than zero"
                    )
                if values["volume"] < 0:
                    raise DataValidationError(f"row {row_number}: volume cannot be negative")
                if (
                    values["high"] < max(values["open"], values["close"])
                    or values["low"] > min(values["open"], values["close"])
                    or values["high"] < values["low"]
                ):
                    raise DataValidationError(
                        f"row {row_number}: inconsistent OHLC high/low values"
                    )

                bars.append(Bar(timestamp=timestamp, **values))
                previous_time = parsed_time
                previous_awareness = is_aware

            if not bars:
                raise DataValidationError("CSV has a header but no market-data rows")
            return bars
        except csv.Error as exc:
            raise DataValidationError(f"invalid CSV data: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise DataValidationError(f"CSV file is not valid UTF-8: {exc}") from exc


def _validate_parameters(
    starting_cash: float,
    allocation: float,
    max_allocation: float,
    fee_rate: float,
    slippage_rate: float,
    drawdown_limit: float | None,
    fast_window: int,
    slow_window: int,
) -> None:
    if not math.isfinite(starting_cash) or starting_cash <= 0:
        raise ValueError("starting cash must be a finite number greater than zero")
    if not math.isfinite(allocation) or not 0 < allocation <= 1:
        raise ValueError("allocation must be greater than 0 and at most 1")
    if not math.isfinite(max_allocation) or not 0 < max_allocation <= 1:
        raise ValueError("maximum allocation must be greater than 0 and at most 1")
    if not math.isfinite(fee_rate) or not 0 <= fee_rate < 1:
        raise ValueError("fee rate must be at least 0 and less than 1")
    if not math.isfinite(slippage_rate) or not 0 <= slippage_rate < 1:
        raise ValueError("slippage rate must be at least 0 and less than 1")
    if drawdown_limit is not None and (
        not math.isfinite(drawdown_limit) or not 0 < drawdown_limit < 1
    ):
        raise ValueError("maximum drawdown limit must be greater than 0 and less than 1")
    if fast_window < 1 or slow_window < 1:
        raise ValueError("moving-average windows must be positive integers")
    if fast_window >= slow_window:
        raise ValueError("fast window must be smaller than slow window")


def backtest(
    bars: Sequence[Bar],
    *,
    starting_cash: float = 200.0,
    allocation: float = 0.25,
    max_allocation: float = 0.25,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    drawdown_limit: float | None = 0.15,
    fast_window: int = 5,
    slow_window: int = 20,
    start_index: int = 0,
) -> BacktestResult:
    """Run a long-only SMA crossover backtest with next-bar-open fills.

    The allocation is the fraction of available cash committed on each entry.
    Fees and adverse slippage are charged on both buys and sells. A drawdown
    circuit breaker halts new entries at the configured close-marked loss
    threshold and exits an open position at the next bar's open. Gaps can make
    realized losses exceed the threshold. Open positions are marked at final
    close and shown as an open trade. Trade count means completed round trips.
    """
    _validate_parameters(
        starting_cash,
        allocation,
        max_allocation,
        fee_rate,
        slippage_rate,
        drawdown_limit,
        fast_window,
        slow_window,
    )
    if not bars:
        raise ValueError("at least one market-data bar is required")
    if not 0 <= start_index < len(bars):
        raise ValueError("evaluation start index must point to a market-data bar")

    cash = starting_cash
    quantity = 0.0
    entry_time = ""
    entry_price = 0.0
    entry_fee = 0.0
    trades: list[Trade] = []
    equity_curve = [starting_cash]
    pending_signal: str | None = None
    previous_fast: float | None = None
    previous_slow: float | None = None
    fast_sum = 0.0
    slow_sum = 0.0
    peak_equity = starting_cash
    max_drawdown = 0.0
    risk_limit_triggered = False
    close_prices = [bar.close for bar in bars]
    open_prices = [bar.open for bar in bars]

    for index, bar in enumerate(bars):
        if (
            index >= start_index
            and pending_signal == "risk_exit"
            and quantity > 0
        ):
            exit_price = open_prices[index] * (1.0 - slippage_rate)
            exit_fee = quantity * exit_price * fee_rate
            cash += quantity * exit_price - exit_fee
            net_pnl = quantity * (exit_price - entry_price) - entry_fee - exit_fee
            trades.append(
                Trade(
                    status="closed",
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=bar.timestamp,
                    exit_price=exit_price,
                    quantity=quantity,
                    entry_fee=entry_fee,
                    exit_fee=exit_fee,
                    net_pnl=net_pnl,
                )
            )
            quantity = 0.0
            entry_fee = 0.0
        elif (
            index >= start_index
            and not risk_limit_triggered
            and pending_signal == "buy"
            and quantity == 0
        ):
            budget = cash * min(allocation, max_allocation)
            entry_price = open_prices[index] * (1.0 + slippage_rate)
            quantity = budget / (entry_price * (1.0 + fee_rate))
            entry_fee = quantity * entry_price * fee_rate
            cash -= quantity * entry_price + entry_fee
            entry_time = bar.timestamp
        elif (
            index >= start_index
            and pending_signal == "sell"
            and quantity > 0
        ):
            exit_price = open_prices[index] * (1.0 - slippage_rate)
            exit_fee = quantity * exit_price * fee_rate
            cash += quantity * exit_price - exit_fee
            net_pnl = quantity * (exit_price - entry_price) - entry_fee - exit_fee
            trades.append(
                Trade(
                    status="closed",
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=bar.timestamp,
                    exit_price=exit_price,
                    quantity=quantity,
                    entry_fee=entry_fee,
                    exit_fee=exit_fee,
                    net_pnl=net_pnl,
                )
            )
            quantity = 0.0
            entry_fee = 0.0
        pending_signal = None

        if index >= start_index:
            equity = cash + quantity * close_prices[index]
            equity_curve.append(equity)
            peak_equity = max(peak_equity, equity)
            current_drawdown = (
                (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            )
            max_drawdown = max(max_drawdown, current_drawdown)
            if (
                drawdown_limit is not None
                and current_drawdown >= drawdown_limit
                and not risk_limit_triggered
            ):
                risk_limit_triggered = True
                if quantity > 0:
                    pending_signal = "risk_exit"

        fast_sum += close_prices[index]
        slow_sum += close_prices[index]
        if index >= fast_window:
            fast_sum -= close_prices[index - fast_window]
        if index >= slow_window:
            slow_sum -= close_prices[index - slow_window]
        fast_average = fast_sum / fast_window if index + 1 >= fast_window else None
        slow_average = slow_sum / slow_window if index + 1 >= slow_window else None

        if not risk_limit_triggered and fast_average is not None and slow_average is not None:
            if previous_fast is not None and previous_slow is not None:
                if previous_fast <= previous_slow and fast_average > slow_average:
                    pending_signal = "buy"
                elif previous_fast >= previous_slow and fast_average < slow_average:
                    pending_signal = "sell"
            previous_fast = fast_average
            previous_slow = slow_average

    ending_equity = equity_curve[-1]
    if quantity > 0:
        final_bar = bars[-1]
        estimated_exit_price = final_bar.close * (1.0 - slippage_rate)
        estimated_exit_fee = quantity * estimated_exit_price * fee_rate
        open_net_pnl = (
            quantity * (estimated_exit_price - entry_price)
            - entry_fee
            - estimated_exit_fee
        )
        trades.append(
            Trade(
                status="open",
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=None,
                exit_price=None,
                quantity=quantity,
                entry_fee=entry_fee,
                exit_fee=0.0,
                net_pnl=open_net_pnl,
            )
        )

    return BacktestResult(
        starting_cash=starting_cash,
        ending_equity=ending_equity,
        total_return=(ending_equity / starting_cash) - 1.0,
        max_drawdown=max_drawdown,
        trade_count=sum(trade.status == "closed" for trade in trades),
        trades=tuple(trades),
        risk_limit_triggered=risk_limit_triggered,
    )


def screen_parameters(
    bars: Sequence[Bar],
    *,
    starting_cash: float,
    allocation: float,
    max_allocation: float,
    fee_rate: float,
    slippage_rate: float,
    drawdown_limit: float | None,
    max_window: int,
    train_fraction: float = 0.7,
) -> tuple[int, int, BacktestResult, BacktestResult, BacktestResult, BacktestResult]:
    """Select SMA parameters on an earlier training segment and test once later."""
    if not math.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError("training fraction must be greater than 0 and less than 1")
    if max_window < 3:
        raise ValueError("maximum search window must be at least 3")
    split = int(len(bars) * train_fraction)
    if split < max_window + 2 or len(bars) - split < 2:
        raise ValueError(
            f"not enough rows for this comparison: need at least "
            f"{max_window + 2} training rows and 2 validation rows"
        )

    ranked = rank_parameters(
        bars[:split],
        starting_cash=starting_cash,
        allocation=allocation,
        max_allocation=max_allocation,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        drawdown_limit=drawdown_limit,
        max_window=max_window,
    )
    selected = ranked[0]
    chosen_fast = selected.fast_window
    chosen_slow = selected.slow_window
    training_result = selected.training_result
    validation_result = backtest(
        bars,
        starting_cash=starting_cash,
        allocation=allocation,
        max_allocation=max_allocation,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        drawdown_limit=drawdown_limit,
        fast_window=chosen_fast,
        slow_window=chosen_slow,
        start_index=split,
    )

    baseline_train = backtest(
        bars[:split],
        starting_cash=starting_cash,
        allocation=allocation,
        max_allocation=max_allocation,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        drawdown_limit=drawdown_limit,
        fast_window=5,
        slow_window=20,
    )
    baseline_validation = backtest(
        bars,
        starting_cash=starting_cash,
        allocation=allocation,
        max_allocation=max_allocation,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        drawdown_limit=drawdown_limit,
        fast_window=5,
        slow_window=20,
        start_index=split,
    )
    return (
        chosen_fast,
        chosen_slow,
        training_result,
        validation_result,
        baseline_train,
        baseline_validation,
    )


def rank_parameters(
    bars: Sequence[Bar],
    *,
    starting_cash: float,
    allocation: float,
    max_allocation: float,
    fee_rate: float,
    slippage_rate: float,
    drawdown_limit: float | None,
    max_window: int,
) -> list[ParameterScore]:
    """Rank by average return across expanding, chronological training folds."""
    if max_window < 3:
        raise ValueError("maximum search window must be at least 3")
    warmup = min(max_window + 1, max(3, len(bars) // 2))
    fold_count = min(3, (len(bars) - warmup) // 2)
    if fold_count < 1:
        raise ValueError(
            "not enough training rows for walk-forward scoring; provide more data "
            "or reduce --max-window"
        )

    fold_boundaries = [
        (
            warmup + (len(bars) - warmup) * fold_index // fold_count,
            warmup + (len(bars) - warmup) * (fold_index + 1) // fold_count,
        )
        for fold_index in range(fold_count)
    ]
    candidates: list[ParameterScore] = []
    for fast_window in range(1, max_window):
        for slow_window in range(fast_window + 1, max_window + 1):
            training_result = backtest(
                bars,
                starting_cash=starting_cash,
                allocation=allocation,
                max_allocation=max_allocation,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                drawdown_limit=drawdown_limit,
                fast_window=fast_window,
                slow_window=slow_window,
            )
            fold_results = tuple(
                backtest(
                    bars[:fold_end],
                    starting_cash=starting_cash,
                    allocation=allocation,
                    max_allocation=max_allocation,
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    drawdown_limit=drawdown_limit,
                    fast_window=fast_window,
                    slow_window=slow_window,
                    start_index=fold_start,
                )
                for fold_start, fold_end in fold_boundaries
            )
            candidates.append(
                ParameterScore(
                    fast_window=fast_window,
                    slow_window=slow_window,
                    training_result=training_result,
                    fold_results=fold_results,
                )
            )

    within_risk_limit = [
        candidate
        for candidate in candidates
        if drawdown_limit is None
        or candidate.worst_fold_drawdown <= drawdown_limit
    ]
    ranked = within_risk_limit or candidates
    return sorted(
        ranked,
        key=lambda candidate: (
            candidate.mean_fold_return,
            -candidate.worst_fold_drawdown,
            candidate.training_result.total_return,
            -candidate.fast_window,
            -candidate.slow_window,
        ),
        reverse=True,
    )


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _data_age_days(timestamp: str, now: datetime | None = None) -> float:
    latest = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    current = now or datetime.now(latest.tzinfo)
    if (current.utcoffset() is None) != (latest.utcoffset() is None):
        raise ValueError("clock and market-data timestamps must use compatible timezones")
    return (current - latest).total_seconds() / 86_400


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Educational OHLCV CSV backtester for a long-only moving-average "
            "crossover strategy. Signals fill at the next bar's open."
        ),
        epilog=(
            "Simulation only: results are historical and do not predict future "
            "performance or guarantee profits."
        ),
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        nargs="?",
        help="market data CSV (default: market_data.csv)",
    )
    parser.add_argument("--fast-window", type=_positive_int, default=5, help="fast SMA length (default: 5)")
    parser.add_argument("--slow-window", type=_positive_int, default=20, help="slow SMA length (default: 20)")
    parser.add_argument("--starting-cash", type=_positive_float, default=200.0, help="starting cash in USD (default: 200)")
    parser.add_argument("--allocation", type=_positive_float, default=0.25, help="requested fraction of available cash used per entry (default: 0.25)")
    parser.add_argument("--max-allocation", type=_positive_float, default=0.25, help="hard cap on cash committed per entry (default: 0.25)")
    parser.add_argument("--fee-rate", type=_positive_float, default=0.001, help="proportional fee per side, from 0 up to but not including 1 (default: 0.001)")
    parser.add_argument("--slippage-rate", type=_positive_float, default=0.0005, help="adverse execution slippage per side (default: 0.0005)")
    parser.add_argument("--max-drawdown", type=_positive_float, default=0.15, help="halt entries at this close-marked drawdown and exit at the next open (default: 0.15; gaps can exceed it)")
    year_group = parser.add_mutually_exclusive_group()
    year_group.add_argument("--year", type=_positive_int, help="only include rows from this calendar year (CSV must contain the data)")
    year_group.add_argument("--current-year", action="store_true", help="only include rows from the current calendar year")
    parser.add_argument("--max-data-age-days", type=_positive_float, default=7.0, help="age after which market data is reported stale (default: 7)")
    parser.add_argument("--require-recent", action="store_true", help="stop instead of backtesting if the last row is older than --max-data-age-days")
    parser.add_argument("--compare", action="store_true", help="screen SMA pairs on the earlier data and report a separate later holdout")
    parser.add_argument("--max-window", type=_positive_int, default=20, help="largest slow SMA in --compare search (default: 20)")
    parser.add_argument("--symbol", default="BTC-USD", help="Yahoo Finance symbol to download (default: BTC-USD)")
    refresh_group = parser.add_mutually_exclusive_group()
    refresh_group.add_argument("--refresh", action="store_true", help="download current daily bars before running")
    refresh_group.add_argument("--offline", action="store_true", help="do not access the network; use the supplied/local CSV")
    parser.add_argument("--watch-minutes", type=_positive_int, help="repeat refresh and report at this minute interval")
    return parser


def _print_result(result: BacktestResult) -> None:
    print(f"Starting cash: ${result.starting_cash:,.2f}")
    print(f"Ending equity: ${result.ending_equity:,.2f}")
    print(f"Total return:  {result.total_return:.2%}")
    print(f"Max drawdown:  {result.max_drawdown:.2%}")
    print(f"Trade count:   {result.trade_count} completed round trip(s)")
    if result.risk_limit_triggered:
        print(
            "RISK CIRCUIT BREAKER: triggered; no new entries, "
            "any exit is at the next bar open"
        )
    if not result.trades:
        print("Trades: none")
        return

    print("Trades:")
    for number, trade in enumerate(result.trades, start=1):
        exit_details = (
            f"exit {trade.exit_time} @ ${trade.exit_price:.4f}"
            if trade.status == "closed"
            else "still open (marked at final close)"
        )
        print(
            f"  {number}. {trade.status.upper()}: enter {trade.entry_time} "
            f"@ ${trade.entry_price:.4f}; {exit_details}; "
            f"qty {trade.quantity:.8f}; net P/L ${trade.net_pnl:,.2f}"
        )


def _print_comparison(
    bars: Sequence[Bar],
    args: argparse.Namespace,
) -> None:
    chosen_fast, chosen_slow, train, validation, base_train, base_validation = (
        screen_parameters(
            bars,
            starting_cash=args.starting_cash,
            allocation=args.allocation,
            max_allocation=args.max_allocation,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            drawdown_limit=args.max_drawdown,
            max_window=args.max_window,
        )
    )
    split = int(len(bars) * 0.7)
    print("Chronological parameter comparison (not a profit guarantee)")
    print(
        f"Training:   {bars[0].timestamp} through {bars[split - 1].timestamp} "
        f"({split} bars)"
    )
    print(
        f"Validation: {bars[split].timestamp} through {bars[-1].timestamp} "
        f"({len(bars) - split} bars)"
    )
    candidates = rank_parameters(
        bars[:split],
        starting_cash=args.starting_cash,
        allocation=args.allocation,
        max_allocation=args.max_allocation,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        drawdown_limit=args.max_drawdown,
        max_window=args.max_window,
    )
    within_limit_count = sum(
        candidate.worst_fold_drawdown <= args.max_drawdown
        for candidate in candidates
    )
    if within_limit_count:
        print(
            f"Walk-forward ranking: mean return across "
            f"{len(candidates[0].fold_results)} expanding folds; "
            f"candidates limited to {args.max_drawdown:.1%} worst-fold drawdown "
            f"({within_limit_count} candidates passed)"
        )
    else:
        print(
            "WARNING: no candidate stayed within the drawdown limit; "
            "showing the highest-return candidate anyway."
        )
    print(
        "Top 5 walk-forward candidates "
        "(mean fold return; worst fold drawdown; full-training return):"
    )
    for number, candidate in enumerate(candidates[:5], start=1):
        print(
            f"  {number}. SMA({candidate.fast_window},{candidate.slow_window}): "
            f"{candidate.mean_fold_return:.2%}; "
            f"{candidate.worst_fold_drawdown:.2%}; "
            f"{candidate.training_result.total_return:.2%}"
        )
    print(
        "Parameters selected by walk-forward mean return; ties prefer lower "
        "worst-fold drawdown, then higher full-training return."
    )
    print(f"  Selected SMA({chosen_fast},{chosen_slow}) training:")
    _print_result(train)
    print(f"  Selected SMA({chosen_fast},{chosen_slow}) validation (fresh ${args.starting_cash:,.2f}):")
    _print_result(validation)
    print("  Default SMA(5,20) validation baseline:")
    _print_result(base_validation)
    print(
        f"  Default SMA(5,20) training return: {base_train.total_return:.2%} "
        f"(not used to choose the validation result)"
    )


def _report_data_coverage(
    bars: Sequence[Bar],
    *,
    max_age_days: float,
    require_recent: bool,
    parser: argparse.ArgumentParser,
) -> None:
    latest = bars[-1].timestamp
    age_days = _data_age_days(latest)
    print(f"Data coverage: {bars[0].timestamp} through {latest} ({len(bars)} bars)")
    if age_days < 0:
        parser.error(f"latest data timestamp is {-age_days:.1f} days in the future")
    if age_days > max_age_days:
        message = (
            f"market data is {age_days:.1f} days old "
            f"(freshness limit: {max_age_days:g} days); results are historical, "
            "not current market conditions"
        )
        if require_recent:
            parser.error(message)
        print(f"WARNING: {message}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from market_data import MarketDataDownloadError, download_daily_ohlcv

        _validate_parameters(
            args.starting_cash,
            args.allocation,
            args.max_allocation,
            args.fee_rate,
            args.slippage_rate,
            args.max_drawdown,
            args.fast_window,
            args.slow_window,
        )
        if args.offline and args.watch_minutes:
            raise ValueError("--watch-minutes cannot be used with --offline")
        auto_refresh = args.csv_file is None
        csv_path = args.csv_file or Path("market_data.csv")
        if args.watch_minutes and not (auto_refresh or args.refresh):
            raise ValueError(
                "--watch-minutes requires automatic refresh; omit the CSV path "
                "or use --refresh"
            )
        while True:
            if args.refresh or (auto_refresh and not args.offline):
                try:
                    bars_downloaded = download_daily_ohlcv(args.symbol, csv_path)
                except MarketDataDownloadError as exc:
                    if not args.watch_minutes:
                        raise
                    print(
                        f"ERROR: market-data refresh failed: {exc}. "
                        f"Retrying in {args.watch_minutes} minute(s).",
                        file=sys.stderr,
                    )
                    sleep(args.watch_minutes * 60)
                    continue
                print(
                    f"Downloaded {len(bars_downloaded)} completed daily bars "
                    f"for {args.symbol.upper()} to {csv_path}"
                )
            bars = load_bars(csv_path)
            requested_year = args.year
            if args.current_year:
                requested_year = datetime.now().year
            if requested_year is not None:
                filtered_bars = []
                for bar in bars:
                    parsed_time = datetime.fromisoformat(bar.timestamp.replace("Z", "+00:00"))
                    if parsed_time.year == requested_year:
                        filtered_bars.append(bar)
                bars = filtered_bars
                if not bars:
                    raise DataValidationError(
                        f"CSV contains no rows for {requested_year}; provide local OHLCV data "
                        "for that year or omit the year filter"
                    )
            _report_data_coverage(
                bars,
                max_age_days=args.max_data_age_days,
                require_recent=args.require_recent,
                parser=parser,
            )
            effective_allocation = min(args.allocation, args.max_allocation)
            print(
                f"Position limit: {effective_allocation:.0%} per entry "
                f"(max ${args.starting_cash * effective_allocation:,.2f} "
                "of starting cash; no guarantee against larger losses)"
            )
            print(
                f"Assumptions: {args.slippage_rate:.3%} adverse slippage per side; "
                f"drawdown circuit breaker {args.max_drawdown:.1%}"
            )
            if args.compare:
                _print_comparison(bars, args)
            else:
                result = backtest(
                    bars,
                    starting_cash=args.starting_cash,
                    allocation=args.allocation,
                    max_allocation=args.max_allocation,
                    fee_rate=args.fee_rate,
                    slippage_rate=args.slippage_rate,
                    drawdown_limit=args.max_drawdown,
                    fast_window=args.fast_window,
                    slow_window=args.slow_window,
                )
                _print_result(result)
            if not args.watch_minutes:
                break
            print(
                f"Next refresh in {args.watch_minutes} minute(s). "
                "Press Ctrl+C to stop."
            )
            sleep(args.watch_minutes * 60)
    except (DataValidationError, MarketDataDownloadError, ValueError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
