# Legacy EuroMillions VBA Model Review

## Workbook structure inferred from VBA

The workbook logic references the following sheets: `All`, `Mains`, `Stars`, `Two`, `Distrib`, `Data`, `MainsD`, `Next Best`, `Test`, `Statistic`, and `Volatility`.

- `All` appears to hold draw history with columns: date/id, five main numbers, and two stars.
- `Mains` and `Stars` store delay histories and percentile-like delay metrics.
- `Two` stores pair frequency and pair delay histories.
- `Distrib` and `Data` provide scoring lookup tables and constants.
- `Next Best` is the top-ranked combinations output table.

## Key VBA procedures and behavior

- `CountDelaysMains` and `CountDelaysStars`: compute inter-arrival gaps and delay percentiles for each number/star.
- `CountTwo` and `ReCountTwo`: count main-number pair co-occurrence frequencies.
- `CountDelaysTwo`: compute delays for each tracked pair.
- `FillMainsD`: transforms delay history into a matrix used for recency shape analysis.
- `Score`, `Score_Backup`, `Score_Old`: weighted scoring variants combining frequency, position, sum, parity, and gap-related factors.
- `NextBest`: brute-force scans all `C(50,5)` combinations and keeps top candidates.
- `UpdateAll`: orchestration macro executing the full pipeline.
- `TurnAround` and `TurnAroundS`: trend/turning-point style metric over delay sequences.

## Reuse in the Python redesign

The following ideas are retained:

- Number and star delay statistics.
- Rolling and lifetime frequencies.
- Pair frequency and pair delay features.
- Sum/parity/gap-based shape scoring.
- Ranked top-candidate generation.

## Redesign decisions

- Replace worksheet mutation with typed, testable pipeline steps.
- Replace nested VBA loops with vectorized pandas/numpy operations where possible.
- Use SQLite persistence for deterministic snapshots and incremental updates.
- Enforce leakage-safe walk-forward evaluation.
- Separate baseline random benchmarking from statistical models.

## Known VBA risks and issues

- Heavy nested loops and cell-by-cell writes are slow and brittle.
- Hardcoded sheet/column references reduce maintainability.
- Global mutable arrays and workbook state reduce reproducibility.
- `Score*` parity counting likely contains a bug (`s4 = s + 1` instead of incrementing `s4`).
- Manual/automatic calculation toggles can leave workbook state inconsistent on errors.
