"""
BreastGuard AI — Exploratory Data Analysis & Statistical Profiler
Provides automated statistical hypothesis testing, effect size computations,
distribution analysis, and multicollinearity diagnostics for the WDBC dataset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.data.loader import DataBundle

logger = logging.getLogger(__name__)


@dataclass
class FeatureComparison:
    """Statistical differentiation metrics for a single feature."""
    feature: str
    mean_malignant: float
    std_malignant: float
    mean_benign: float
    std_benign: float
    t_stat: float
    t_pvalue: float
    u_stat: float
    u_pvalue: float
    cohens_d: float
    significant_p001: bool


@dataclass
class EDAResults:
    """Container holding full EDA profile."""
    summary_df: pd.DataFrame
    comparison_df: pd.DataFrame
    top_differentiating_features: list[str]
    correlation_matrix: pd.DataFrame
    high_correlation_pairs: list[dict]
    class_balance: dict[str, int]


def calculate_cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two independent groups.
    d = (mean1 - mean2) / pooled_std
    """
    n1, n2 = len(x1), len(x2)
    s1_sq, s2_sq = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x1) - np.mean(x2)) / pooled_std)


def analyze_dataset(bundle: DataBundle) -> EDAResults:
    """
    Perform deep statistical profiling on the WDBC dataset.

    Parameters
    ----------
    bundle : DataBundle
        Loaded and validated dataset bundle.

    Returns
    -------
    EDAResults
        Comprehensive statistical summary, comparisons, and correlations.
    """
    logger.info("Computing EDA statistical profiles ...")
    df = bundle.X_df.copy()
    y = bundle.y

    # Class breakdown: 0 = Malignant, 1 = Benign
    n_mal = int((y == 0).sum())
    n_ben = int((y == 1).sum())
    class_balance = {"Malignant": n_mal, "Benign": n_ben}

    # Summary table
    summary_df = df.describe().T
    summary_df["skew"] = df.skew()
    summary_df["kurtosis"] = df.kurtosis()

    # Feature comparison between classes
    comparisons: list[FeatureComparison] = []
    mal_mask = (y == 0)
    ben_mask = (y == 1)

    for feat in bundle.feature_names:
        x_mal = df.loc[mal_mask, feat].values
        x_ben = df.loc[ben_mask, feat].values

        # Independent t-test (Welch's t-test for unequal variances)
        t_stat, t_pval = stats.ttest_ind(x_mal, x_ben, equal_var=False)

        # Non-parametric Mann-Whitney U test
        u_stat, u_pval = stats.mannwhitneyu(x_mal, x_ben, alternative="two-sided")

        # Effect size
        c_d = calculate_cohens_d(x_mal, x_ben)

        comparisons.append(
            FeatureComparison(
                feature=feat,
                mean_malignant=float(np.mean(x_mal)),
                std_malignant=float(np.std(x_mal, ddof=1)),
                mean_benign=float(np.mean(x_ben)),
                std_benign=float(np.std(x_ben, ddof=1)),
                t_stat=float(t_stat),
                t_pvalue=float(t_pval),
                u_stat=float(u_stat),
                u_pvalue=float(u_pval),
                cohens_d=c_d,
                significant_p001=(u_pval < 0.001),
            )
        )

    comp_df = pd.DataFrame([vars(c) for c in comparisons])
    # Rank by magnitude of effect size |Cohen's d|
    comp_df["abs_cohens_d"] = comp_df["cohens_d"].abs()
    comp_df = comp_df.sort_values(by="abs_cohens_d", ascending=False).reset_index(drop=True)

    top_differentiating = comp_df["feature"].head(10).tolist()

    # Correlation matrix & multicollinearity search
    corr_matrix = df.corr(method="pearson")
    pairs: list[dict] = []
    features = bundle.feature_names

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            r = corr_matrix.iloc[i, j]
            if abs(r) >= 0.85:
                pairs.append({
                    "feature_1": features[i],
                    "feature_2": features[j],
                    "correlation": float(r),
                    "abs_corr": float(abs(r)),
                })

    pairs = sorted(pairs, key=lambda x: x["abs_corr"], reverse=True)

    logger.info(
        "EDA complete. Top 3 differentiating features: %s. High correlation pairs (r>=0.85): %d",
        top_differentiating[:3],
        len(pairs),
    )

    return EDAResults(
        summary_df=summary_df,
        comparison_df=comp_df,
        top_differentiating_features=top_differentiating,
        correlation_matrix=corr_matrix,
        high_correlation_pairs=pairs,
        class_balance=class_balance,
    )

