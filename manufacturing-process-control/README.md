# Manufacturing Process Analysis

Statistical Process Control (SPC) analysis using SQL to monitor manufacturing quality.

## Goal
Identify when product measurements fall outside acceptable control limits, triggering process adjustments.

## SQL Concepts
- **Window functions**: `AVG()`, `STDDEV()`, `ROW_NUMBER()` with `OVER` clause
- **Sliding windows**: `ROWS BETWEEN 4 PRECEDING AND CURRENT ROW`
- **Partitioning**: Analysis grouped by machine operator
- **CASE statements**: Flagging out-of-control measurements
