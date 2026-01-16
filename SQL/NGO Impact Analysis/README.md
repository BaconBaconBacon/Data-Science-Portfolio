# NGO Impact Analysis

Analysis of donation patterns and assignment impact for GoodThought NGO using PostgreSQL.

## Goal
Identify top-performing assignments by donation value and regional impact scores.

## SQL Concepts
- **Common Table Expressions (CTEs)**: Modular query building
- **JOINs**: Linking assignments, donations, and donors
- **Aggregations**: `SUM()`, `COUNT()` with `GROUP BY` and `HAVING`
- **Window functions**: `ROW_NUMBER()` for ranking within partitions
