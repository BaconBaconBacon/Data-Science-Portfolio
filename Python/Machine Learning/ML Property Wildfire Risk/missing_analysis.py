"""Missing data analysis for determining imputation strategy.

Analyzes the missingness mechanism (MCAR/MAR) for each feature to select
the appropriate imputation method:
- MCAR (Missing Completely At Random): Use SimpleImputer (median)
- MAR (Missing At Random): Use IterativeImputer

Detection method: Correlate each feature's missingness indicator with all
other features. High correlation suggests MAR (missingness depends on
observed values).
"""

import numpy as np
import pandas as pd
import load_census


def analyze_missingness(
    df: pd.DataFrame,
    threshold: float = 0.1,
    min_samples: int = 100,
) -> dict:
    """
    Analyze missing data mechanism for each feature with missing values.

    For each feature, creates a binary missingness indicator and correlates
    it with all other features. If the maximum absolute correlation exceeds
    the threshold, the feature is classified as MAR; otherwise MCAR.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.
    threshold : float
        Correlation threshold for MAR classification. Features with
        missingness correlation > threshold are classified as MAR.
    min_samples : int
        Minimum non-missing samples required to compute correlation.

    Returns
    -------
    dict
        Mapping of feature name -> {
            'mechanism': 'MCAR' | 'MAR',
            'max_corr': float,
            'corr_with': str (feature with highest correlation),
            'missing_pct': float (percentage of missing values)
        }
    """
    results = {}
    features_with_missing = df.columns[df.isna().any()].tolist()

    if not features_with_missing:
        return results

    # Pre-compute which features have enough non-missing values
    valid_features = [c for c in df.columns if df[c].notna().sum() >= min_samples]

    for feat in features_with_missing:
        # Create missingness indicator (1 = missing, 0 = present)
        missing_indicator = df[feat].isna().astype(int)

        # Skip if all or none are missing
        if missing_indicator.sum() == 0 or missing_indicator.sum() == len(df):
            continue

        max_corr = 0.0
        max_corr_feat = None

        # Correlate with all other valid features
        for other_feat in valid_features:
            if other_feat == feat:
                continue

            # Use rows where other_feat is not missing
            mask = df[other_feat].notna()
            if mask.sum() < min_samples:
                continue

            # Skip zero-variance columns (would cause division by zero in correlation)
            other_values = df[other_feat][mask]
            indicator_subset = missing_indicator[mask]
            if other_values.std() < 1e-10 or indicator_subset.std() < 1e-10:
                continue

            try:
                corr = abs(missing_indicator[mask].corr(other_values))
                if pd.notna(corr) and corr > max_corr:
                    max_corr = corr
                    max_corr_feat = other_feat
            except Exception:
                continue

        # Classify mechanism based on correlation threshold
        mechanism = "MAR" if max_corr > threshold else "MCAR"

        results[feat] = {
            "mechanism": mechanism,
            "max_corr": max_corr,
            "corr_with": max_corr_feat,
            "missing_pct": df[feat].isna().mean() * 100,
        }

    return results


def print_missingness_report(analysis: dict, top_n: int = 5) -> None:
    """
    Print a summary report of missingness analysis.

    Parameters
    ----------
    analysis : dict
        Output from analyze_missingness().
    top_n : int
        Number of top MAR features to display.
    """
    if not analysis:
        print("\nNo features with missing values found.")
        return

    mcar = [f for f, v in analysis.items() if v["mechanism"] == "MCAR"]
    mar = [f for f, v in analysis.items() if v["mechanism"] == "MAR"]

    print(f"\nMissingness Analysis Report:")
    print(f"  MCAR features: {len(mcar)} (will use median imputation)")
    print(f"  MAR features:  {len(mar)} (will use iterative imputation)")

    # Summary statistics
    if analysis:
        missing_pcts = [v["missing_pct"] for v in analysis.values()]
        print(f"\n  Missing data summary:")
        print(f"    Features with missing: {len(analysis)}")
        print(f"    Avg missing %: {np.mean(missing_pcts):.1f}%")
        print(f"    Max missing %: {np.max(missing_pcts):.1f}%")

    if mar:
        print(
            f"\n  Top {min(top_n, len(mar))} MAR features (missingness correlates with other features):"
        )
        sorted_mar = sorted(mar, key=lambda f: analysis[f]["max_corr"], reverse=True)[
            :top_n
        ]
        for f in sorted_mar:
            v = analysis[f]
            corr_with = v["corr_with"] or "N/A"
            readable_f = load_census.census_code_to_label(f)
            readable_corr = (
                load_census.census_code_to_label(corr_with)
                if corr_with != "N/A"
                else "N/A"
            )
            print(
                f"    {readable_f}: {v['missing_pct']:.1f}% missing, "
                f"corr={v['max_corr']:.3f} with {readable_corr}"
            )


def get_column_lists(
    analysis: dict,
    all_columns: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Categorize columns into MCAR, MAR, and complete (no missing).

    Parameters
    ----------
    analysis : dict
        Output from analyze_missingness().
    all_columns : list[str]
        All column names in the DataFrame.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        (mcar_cols, mar_cols, complete_cols)
    """
    mcar_cols = [f for f, v in analysis.items() if v["mechanism"] == "MCAR"]
    mar_cols = [f for f, v in analysis.items() if v["mechanism"] == "MAR"]
    missing_cols = set(mcar_cols + mar_cols)
    complete_cols = [c for c in all_columns if c not in missing_cols]

    return mcar_cols, mar_cols, complete_cols
