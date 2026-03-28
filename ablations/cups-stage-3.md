⏺ Here are all 9 validation results so far:

┌──────┬───────┬───────────┬──────────┬───────┐
│ Eval │  PQ   │ PQ_things │ PQ_stuff │ mIoU  │
├──────┼───────┼───────────┼──────────┼───────┤
│ 1    │ 26.69 │ 22.83     │ 28.96    │ 42.1% │
├──────┼───────┼───────────┼──────────┼───────┤
│ 2    │ 27.67 │ 27.20     │ 27.94    │ 41.9% │
├──────┼───────┼───────────┼──────────┼───────┤
│ 3    │ 25.78 │ 20.31     │ 29.00    │ 41.5% │
├──────┼───────┼───────────┼──────────┼───────┤
│ 4    │ 27.19 │ 26.12     │ 27.82    │ 41.5% │
├──────┼───────┼───────────┼──────────┼───────┤
│ 5    │ 26.74 │ 23.29     │ 28.78    │ 41.5% │
├──────┼───────┼───────────┼──────────┼───────┤
│ 6    │ 27.14 │ 26.22     │ 27.69    │ 41.2% │
├──────┼───────┼───────────┼──────────┼───────┤
│ 7    │ 26.65 │ 23.48     │ 28.51    │ 41.7% │
├──────┼───────┼───────────┼──────────┼───────┤
│ 8    │ 26.01 │ 23.61     │ 27.42    │ 40.7% │
├──────┼───────┼───────────┼──────────┼───────┤
│ 9    │ 26.60 │ 23.78     │ 28.27    │ 41.7% │
└──────┴───────┴───────────┴──────────┴───────┘

You're right — it's plateauing. Oscillating between 25.8-27.7 with no upward trend after eval #2.

But there's a silver lining: PQ_things peaked at 27.20 (eval #2), way up from Stage-2's 22.57. If we combine that checkpoint's instances with UNet
semantics (PQ_stuff=35.04), the estimated combined PQ would be roughly ~31.7 — well above CUPS's 27.8.

Currently at iteration 259/869 (~3 hours in), still in Round 1 of 3. Want me to kill it and use the best checkpoint from eval #2, or let it continue
through all 3 rounds?