# Educational Paper-Trading Backtester

This small Python command-line project simulates a long-only moving-average
crossover strategy using OHLCV CSV data. It uses only the Python standard
library, can optionally download public daily market data, and has no broker
connection. It cannot place trades.

## Requirements and run

Python 3.10 or newer is recommended.

On Windows, double-click `start.bat` to fetch completed daily BTC-USD candles,
run the current-year comparison, and refresh/report hourly (requires internet
access and Python 3). Press Ctrl+C to stop. From a terminal, use:

```text
python -m unittest -v
python paper_trading.py sample_data.csv --compare
python paper_trading.py --current-year --require-recent --compare
python paper_trading.py --symbol AAPL --current-year --require-recent --compare
python paper_trading.py path/to/market.csv --year 2026 --compare
python paper_trading.py path/to/market.csv --starting-cash 200 --allocation 0.5 --max-allocation 0.25 --fee-rate 0.001 --slippage-rate 0.0005
python paper_trading.py --offline market_data.csv --current-year --compare
python paper_trading.py --help
```

The default starting capital is **$200**. The default requested allocation
and hard per-entry cap are both **25%**, so at most $50 of a $200 account is
spent on a position including its entry fee. `--allocation` cannot override
`--max-allocation`; raising the cap requires explicitly changing that flag.
This limits concentration, not losses: a severe gap can still lose most or all
of the amount invested. A default **15% close-marked drawdown circuit breaker**
halts new entries and schedules an open-position exit at the next bar's open;
the realized loss may exceed 15% due to gaps or execution costs. Set
`--max-drawdown` to adjust it (must be greater than 0 and less than 1).
The backtest applies adverse slippage of **0.05% per side** by default, in
addition to fees; change it with `--slippage-rate`. These are explicit assumptions,
not guarantees of real execution. If no CSV path is given, the program
downloads current-calendar-year daily OHLCV from Yahoo Finance's public chart
endpoint into `market_data.csv`, replacing that file atomically only after
validating the response. The default symbol is `BTC-USD`; set `--symbol` to
another Yahoo Finance symbol. Only completed daily candles are included.
`--refresh` explicitly refreshes even when a CSV path is provided. `--offline`
prevents network access and reads the supplied path or `market_data.csv`.
`--watch-minutes 60` repeats the download/report every hour until Ctrl+C.

`--year` filters data to a specific calendar year; `--current-year` selects
this computer's current calendar year; omit both to use all supplied rows.
The report prints the date range and age of data. Data older than seven days
is flagged by default; `--require-recent` stops instead of calculating on
stale data. The bundled sample is illustrative, not current market data.

CSV files must have a header containing `timestamp,open,high,low,close,volume`.
Column names are case-insensitive. Timestamps must be ISO 8601 dates or
datetimes (for example, `2024-01-01` or `2024-01-01T09:30:00Z`) and rows must
be strictly chronological. Prices must be finite and positive, volume must be
finite and nonnegative, and each row's high/low must be consistent with its
open and close. See `sample_data.csv` for a complete example.

## Strategy and simulator assumptions

The fast and slow simple moving averages (SMAs) use closing prices. A bullish
crossover occurs when the fast SMA moves from at or below the slow SMA to above
it; a bearish crossover occurs on the reverse. A crossover is only known after
that bar closes. Therefore, its order is filled at the **next bar's open**,
never at the close that generated the signal. A signal on the final bar cannot
be filled because no subsequent bar is available.

The strategy holds at most one long position and does not short, use leverage,
or add to an existing position. On entry it commits the lesser of the requested
fraction and the hard cap of available cash, accounting for the entry fee;
fractional units are allowed. Adverse slippage is applied to each fill (buy
above and sell below the bar open); the proportional fee is charged on both
entry and exit. The drawdown breaker is assessed from close-marked equity and,
once triggered, blocks future entries and exits at the next bar open. A gap
through that exit price can exceed the threshold; if the trigger occurs on the
last bar, no next open is available and the position remains open. Remaining
positions are marked to the final close for ending equity; their displayed unrealized
P/L estimates a liquidation fee. Reported trade count includes completed
round trips only, while an open final position is still listed in the trades.
Maximum drawdown is calculated from the close-marked equity curve, including
initial cash.

With `--compare`, the program searches fast/slow SMA pairs up to `--max-window`
(default 20). It evaluates each candidate on up to three expanding walk-forward
folds inside the earlier 70%, with indicators warmed only from preceding
historical closes. It ranks by mean fold return after fee and slippage, then
worst fold drawdown and full-training return. Candidates exceeding the
configured drawdown limit on any fold are excluded if any candidate passes;
otherwise the report warns and shows the least-bad available ranking. The top
pair and default SMA(5,20) are then evaluated separately on the later 30%, each
starting with fresh capital. This final holdout is not used for selection.
Walk-forward scoring reduces reliance on one in-sample period but cannot
eliminate overfitting; trying many pairs can still overfit. Historical net
returns do not estimate or ensure future profit. Provide enough rows for folds
and the final holdout.

The Windows launcher refreshes daily BTC-USD data on startup and every hour,
filters to this calendar year, and refuses data older than seven days. The
provider is an unofficial best-effort public endpoint: availability, coverage,
accuracy, and terms can change. Verify data independently before relying on it.
This is daily historical data, not a live tick feed.

## Important risk note

This is an educational simulator, not financial advice or a recommendation.
Backtests are sensitive to data quality, assumptions, fees, and market
conditions; historical results do not predict future performance. Losses are
possible, and no strategy, cap, stop, or result guarantees profits or limits
losses to a fixed amount.

The CLI has an optional daily market-data download as described above; this is
not a live tick feed and may be delayed, incomplete, unavailable, or inaccurate.
The bundled sample remains historical demonstration data.
