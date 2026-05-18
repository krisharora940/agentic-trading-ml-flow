# Broker And Platform Decisions

Use this reference when selecting or implementing broker, exchange, or managed platform integrations.

## Unified framework advantage

The common self-inflicted failure is technical divergence between backtest and live systems. Feature calculation differences, timing assumptions, and data edge cases make live performance diverge from research.

Use one strategy code path across:

- Backtest and validation.
- Paper trading.
- Live trading.

Adapters should hide data and execution sources. Strategy logic should stay portable.

## IBKR

IBKR is the stronger path when routing quality, asset coverage, and professional execution controls matter.

Core requirements:

- TWS or IB Gateway connection mode.
- Heartbeat requests for idle connection detection.
- Exponential backoff reconnection.
- Position callbacks.
- Account value callbacks.
- Execution report handling.
- Error-code handling.

Relevant order types:

- Market.
- Limit.
- Stop.
- Broker-supported routing such as SmartRouting.

IBKR Pro SmartRouting matters because broker routing architecture can explain more execution cost variation than commission schedules.

## Alpaca

Alpaca is usually the easier early paper/live path for US stocks and ETFs.

Use it when:

- Fast paper trading is the priority.
- REST and streaming API simplicity matters.
- Strategy scope fits supported assets and order types.

Do not ignore realized execution cost. Commission-free trading does not remove spread, routing, fill, or latency cost.

Crypto support can be useful, but venue eligibility and jurisdiction constraints must be verified before strategy development.

## Direct Crypto Exchanges

For Binance, Bybit, OKX, Deribit, and similar venues:

- Verify geographic access.
- Verify account eligibility.
- Model funding, leverage, liquidation, and contract specs.
- Keep exchange adapters isolated.
- Expect venue-specific market data behavior.

## QuantConnect And Managed Platforms

Managed platforms suit:

- Retail-scale operation.
- Rapid prototyping.
- Limited engineering bandwidth.
- Strategies that fit platform constraints.

Self-hosted systems suit:

- Custom data handling.
- Portability.
- Tighter operational control.
- Sensitive intellectual property.

LEAN is a hybrid option because it can run cloud, local, or self-hosted. Plan migration paths early if platform lock-in would matter.
