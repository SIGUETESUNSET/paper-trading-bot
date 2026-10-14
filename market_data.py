"""Daily OHLCV download from Yahoo Finance's public chart endpoint.

This is a best-effort educational data source, not an official market-data
service. No credentials or broker connections are used.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from paper_trading import Bar, DataValidationError


class MarketDataDownloadError(ValueError):
    """Raised when current daily OHLCV data cannot be fetched or verified."""


def parse_chart_response(
    payload: dict[str, Any],
    *,
    symbol: str,
    today: date | None = None,
) -> list[Bar]:
    """Validate Yahoo chart JSON and keep completed daily bars only."""
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        raise MarketDataDownloadError("market-data response has no chart object")
    if chart.get("error"):
        error = chart["error"]
        description = error.get("description", "provider returned an unspecified error")
        raise MarketDataDownloadError(f"data provider rejected {symbol}: {description}")
    results = chart.get("result")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        raise MarketDataDownloadError(f"provider returned no daily data for {symbol}")

    result = results[0]
    metadata = result.get("meta") or {}
    response_symbol = metadata.get("symbol")
    if response_symbol and response_symbol.upper() != symbol.upper():
        raise MarketDataDownloadError(
            f"provider returned {response_symbol} instead of requested {symbol}"
        )
    currency = metadata.get("currency")
    if currency and currency.upper() != "USD":
        raise MarketDataDownloadError(
            f"{symbol} is quoted in {currency}, but starting cash is denominated in USD"
        )
    timezone_name = metadata.get("exchangeTimezoneName")
    if not timezone_name:
        raise MarketDataDownloadError("provider response is missing exchange timezone")
    try:
        exchange_timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise MarketDataDownloadError(
            f"provider returned unknown exchange timezone {timezone_name!r}"
        ) from exc

    timestamps = result.get("timestamp")
    quotes = (result.get("indicators") or {}).get("quote")
    if not isinstance(timestamps, list) or not isinstance(quotes, list) or not quotes:
        raise MarketDataDownloadError(f"provider returned incomplete OHLCV data for {symbol}")
    quote_data = quotes[0]
    if not isinstance(quote_data, dict):
        raise MarketDataDownloadError(f"provider returned malformed OHLCV data for {symbol}")
    fields = ("open", "high", "low", "close", "volume")
    arrays = {name: quote_data.get(name) for name in fields}
    if any(not isinstance(values, list) for values in arrays.values()):
        raise MarketDataDownloadError(f"provider returned incomplete OHLCV fields for {symbol}")

    current_date = today or datetime.now(exchange_timezone).date()
    bars: list[Bar] = []
    for index, timestamp in enumerate(timestamps):
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise MarketDataDownloadError("provider returned an invalid candle timestamp")
        try:
            candle_time = datetime.fromtimestamp(timestamp, exchange_timezone)
        except (OverflowError, OSError, ValueError) as exc:
            raise MarketDataDownloadError(
                "provider returned an out-of-range candle timestamp"
            ) from exc
        if candle_time.date() >= current_date:
            continue
        if any(index >= len(values) or values[index] is None for values in arrays.values()):
            continue

        numbers: dict[str, float] = {}
        for name in fields:
            try:
                number = float(arrays[name][index])
            except (TypeError, ValueError) as exc:
                raise MarketDataDownloadError(
                    f"provider returned a non-numeric {name} value"
                ) from exc
            if not math.isfinite(number):
                raise MarketDataDownloadError(
                    f"provider returned a non-finite {name} value"
                )
            numbers[name] = number

        try:
            bars.append(Bar(timestamp=candle_time.isoformat(), **numbers))
        except TypeError as exc:
            raise MarketDataDownloadError("provider returned malformed OHLCV data") from exc

    if len(bars) < 2:
        raise MarketDataDownloadError(
            f"provider returned only {len(bars)} completed daily bars for {symbol}"
        )

    try:
        return _validate_bars(bars, symbol)
    except DataValidationError as exc:
        raise MarketDataDownloadError(
            f"downloaded data for {symbol} failed OHLCV validation: {exc}"
        ) from exc


def _validate_bars(bars: list[Bar], symbol: str) -> list[Bar]:
    previous: datetime | None = None
    for index, bar in enumerate(bars, start=1):
        current = datetime.fromisoformat(bar.timestamp)
        if previous is not None and current <= previous:
            raise DataValidationError(f"bar {index} is not strictly chronological")
        if any(
            not math.isfinite(value)
            for value in (bar.open, bar.high, bar.low, bar.close, bar.volume)
        ):
            raise DataValidationError(f"bar {index} contains a non-finite value")
        if min(bar.open, bar.high, bar.low, bar.close) <= 0 or bar.volume < 0:
            raise DataValidationError(f"bar {index} contains invalid price or volume")
        if (
            bar.high < max(bar.open, bar.close)
            or bar.low > min(bar.open, bar.close)
            or bar.high < bar.low
        ):
            raise DataValidationError(f"bar {index} has inconsistent high/low")
        previous = current
    if not symbol.strip():
        raise DataValidationError("symbol cannot be empty")
    return bars


def download_daily_ohlcv(
    symbol: str,
    output_path: str | Path,
    *,
    timeout: float = 20.0,
) -> list[Bar]:
    """Fetch current-calendar-year daily bars, excluding today's incomplete bar.

    The target is replaced atomically only after the complete response passes
    validation, so a network/provider error does not corrupt a prior CSV.
    """
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise MarketDataDownloadError("symbol cannot be empty")

    year = datetime.now(timezone.utc).year
    period_start = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())
    period_end = int(datetime.now(timezone.utc).timestamp())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(normalized_symbol, safe='')}?period1={period_start}"
        f"&period2={period_end}&interval=1d&events=history"
    )
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "EducationalPaperTradingSimulator/1.0",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise MarketDataDownloadError(
                    f"data provider returned HTTP status {response.status}"
                )
            payload = json.load(response)
    except HTTPError as exc:
        raise MarketDataDownloadError(
            f"data provider returned HTTP status {exc.code}"
        ) from exc
    except URLError as exc:
        raise MarketDataDownloadError(
            f"could not reach data provider: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise MarketDataDownloadError(
            f"data provider did not respond within {timeout:g} seconds"
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MarketDataDownloadError("data provider returned invalid JSON") from exc
    except OSError as exc:
        raise MarketDataDownloadError(f"market-data request failed: {exc}") from exc

    bars = parse_chart_response(payload, symbol=normalized_symbol)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(("timestamp", "open", "high", "low", "close", "volume"))
            for bar in bars:
                writer.writerow(
                    (
                        bar.timestamp,
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume,
                    )
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
    except OSError as exc:
        raise MarketDataDownloadError(
            f"could not safely write market data to {output}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return bars
