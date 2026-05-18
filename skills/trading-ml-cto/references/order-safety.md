# Order Lifecycle And Safety

Use this reference before implementing live or paper order handling.

## Order State Machine

Represent the order journey explicitly:

```text
signal -> intent -> submitted -> acknowledged -> partial_fill -> filled
                                      |              |
                                      v              v
                                  rejected       canceled
                                      |
                                      v
                                    failed
```

The implementation must tolerate:

- Out-of-order broker messages.
- Partial fills.
- Cancel while fill is in flight.
- Network timeout after submission.
- Duplicate event delivery.
- Broker-side rejection after local submission.

## Idempotency

Every order needs a client order ID.

Use it to:

- Recover after process crash.
- Prevent duplicate order submission.
- Match broker events to local records.
- Reconcile intended vs actual state.

Retry policy must check existing client order ID status before resubmitting.

## Reconciliation

End-of-day reconciliation compares:

- Internal positions vs broker positions.
- Internal orders vs broker orders.
- Internal cash vs broker cash.
- Internal executions vs broker executions.

If reconciliation fails, halt automated trading. Restart only after the mismatch is explained or manually accepted with a logged reason.

## Startup Gates

Before automated trading:

- Verify secrets and auth.
- Verify broker connection.
- Verify clock sync.
- Verify market session.
- Verify data freshness.
- Verify account state.
- Verify open orders.
- Verify risk limits.

## Kill Switches

Use four levels:

1. Pause new signals.
2. Cancel open orders.
3. Flatten positions.
4. Full shutdown.

Each must be tested before live trading.

## Circuit Breakers

Use independent breaker scopes:

- Per-trade validation.
- Per-strategy exposure.
- Portfolio-wide aggregate risk.
- System health.

Software breakers should use CLOSED, OPEN, and HALF_OPEN states with explicit resume criteria.
