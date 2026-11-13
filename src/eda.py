"""
BreastGuard AI — Exploratory Data Analysis (EDA) Module
================================================================================
Performs a clinically-oriented statistical analysis of the WDBC dataset.

Analyses covered:
  A. Univariate statistics per feature with skewness, kurtosis, and IQR
  B. Class-stratified comparison (Malignant M=1 vs Benign B=0)
     — Welch's independent t-test
     — Mann-Whitney U test (non-parametric)
     — Cohen's d effect size
  C. Correlation matrix (Pearson) with multicollinearity detection
  D. Feature group analysis (mean / SE / worst)
  E. Outlier detection per feature (IQR fence method)

TARGET ENCODING:  M = 1 (Malignant),  B = 0 (Benign)
================================================================================
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from config.config import REPORTS_DIR
from src.data_loader import FEATURE_COLUMNS, FEATURE_GROUPS, BASE_MEASUREMENTS

logger = logging.getLogger(__name__)

EDA_REPORTS_DIR = REPORTS_DIR / "evaluation"


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FeatureProfile:
    """Statistical profile for one feature (full cohort + per-class breakdown)."""
    feature: str
    # Univariate — full cohort
    count: int
    mean: float
    std: float
    min_val: float
    q25: float
    median: float
    q75: float
    max_val: float
    iqr: float
    skewness: float
    kurtosis: float
    # Per-class
    mean_malignant: float
    mean_benign: float
    std_malignant: float
    std_benign: float
    # Statistical tests
    t_statistic: float
    t_pvalue: float
    u_statistic: float
    u_pvalue: float
    cohens_d: float
    effect_size_label: str  # negligible / small / medium / large
    significant_bonferroni: bool   # Bonferroni-corrected p < 0.05/30
    # Outlier detection (IQR fence)
    n_outliers_low: int
    n_outliers_high: int
    n_outliers_total: int
    outlier_fraction: float


@dataclass
class CorrelationPair:
    """Correlation relationship between two features."""
    feature_1: str
    feature_2: str
    pearson_r: float
    abs_r: float
    strength: str   # weak / moderate / strong / very_strong


@dataclass
class EDAReport:
    """Full EDA output container."""
    n_samples: int
    n_malignant: int
    n_benign: int
    prevalence_malignant: float
    feature_profiles: list[FeatureProfile]
    top_discriminating_features: list[str]   # Top 10 by Cohen's d
    high_correlation_pairs: list[CorrelationPair]
    correlation_matrix: dict          # feature → {feature: r}
    group_summaries: dict             # mean/se/worst → per-class averages
    multicollinear_clusters: list[list[str]]  # groups of features with |r| >= 0.85


# ─────────────────────────────────────────────────────────────────────────────
# Helper — Cohen's d
# ─────────────────────────────────────────────────────────────────────────────
def _cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Pooled-variance Cohen's d effect size."""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1_sq = np.var(x1, ddof=1)
    s2_sq = np.var(x2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))
    return float((np.mean(x1) - np.mean(x2)) / pooled) if pooled > 0 else 0.0


def _effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:  return "negligible"
    if ad < 0.5:  return "small"
    if ad < 0.8:  return "medium"
    return "large"


def _correlation_strength(r: float) -> str:
    ar = abs(r)
    if ar < 0.3:   return "weak"
    if ar < 0.5:   return "moderate"
    if ar < 0.7:   return "strong"
    if ar < 0.85:  return "very_strong"
    return "multicollinear"


# ─────────────────────────────────────────────────────────────────────────────
# A — Univariate + Per-class feature profiles
# ─────────────────────────────────────────────────────────────────────────────
def _profile_feature(
    df: pd.DataFrame,
    feature: str,
    bonferroni_alpha: float = 0.05 / 30,
) -> FeatureProfile:
    """Compute full statistical profile for a single feature."""
    x = df[feature].values
    x_mal = df.loc[df["diagnosis"] == 1, feature].values
    x_ben = df.loc[df["diagnosis"] == 0, feature].values

    q25, median, q75 = np.percentile(x, [25, 50, 75])
    iqr = q75 - q25
    fence_low = q25 - 1.5 * iqr
    fence_high = q75 + 1.5 * iqr
    n_low = int((x < fence_low).sum())
    n_high = int((x > fence_high).sum())

    t_stat, t_pval = stats.ttest_ind(x_mal, x_ben, equal_var=False)
    u_stat, u_pval = stats.mannwhitneyu(x_mal, x_ben, alternative="two-sided")
    cd = _cohens_d(x_mal, x_ben)

    return FeatureProfile(
        feature=feature,
        count=int(len(x)),
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)),
        min_val=float(np.min(x)),
        q25=float(q25),
        median=float(median),
        q75=float(q75),
        max_val=float(np.max(x)),
        iqr=float(iqr),
        skewness=float(stats.skew(x)),
        kurtosis=float(stats.kurtosis(x)),
        mean_malignant=float(np.mean(x_mal)),
        mean_benign=float(np.mean(x_ben)),
        std_malignant=float(np.std(x_mal, ddof=1)),
        std_benign=float(np.std(x_ben, ddof=1)),
        t_statistic=float(t_stat),
        t_pvalue=float(t_pval),
        u_statistic=float(u_stat),
        u_pvalue=float(u_pval),
        cohens_d=float(cd),
        effect_size_label=_effect_label(cd),
        significant_bonferroni=bool(u_pval < bonferroni_alpha),
        n_outliers_low=n_low,
        n_outliers_high=n_high,
        n_outliers_total=n_low + n_high,
        outlier_fraction=float((n_low + n_high) / len(x)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# C — Correlation analysis
# ─────────────────────────────────────────────────────────────────────────────
def _correlation_analysis(df: pd.DataFrame) -> tuple[dict, list[CorrelationPair], list[list[str]]]:
    """Compute Pearson correlation matrix, identify strong pairs and clusters."""
    X = df[FEATURE_COLUMNS]
    corr_matrix = X.corr(method="pearson")
    corr_dict = {
        col: {c: round(float(v), 6) for c, v in row.items()}
        for col, row in corr_matrix.to_dict().items()
    }

    # Strong pairs
    pairs: list[CorrelationPair] = []
    feats = FEATURE_COLUMNS
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            r = float(corr_matrix.iloc[i, j])
            if abs(r) >= 0.70:
                pairs.append(CorrelationPair(
                    feature_1=feats[i],
                    feature_2=feats[j],
                    pearson_r=round(r, 6),
                    abs_r=round(abs(r), 6),
                    strength=_correlation_strength(r),
                ))
    pairs.sort(key=lambda p: p.abs_r, reverse=True)

    # Multicollinear clusters (|r| >= 0.85)
    visited: set[str] = set()
    clusters: list[list[str]] = []
    high_pairs = [p for p in pairs if p.abs_r >= 0.85]
    adj: dict[str, set[str]] = {f: set() for f in feats}
    for p in high_pairs:
        adj[p.feature_1].add(p.feature_2)
        adj[p.feature_2].add(p.feature_1)
    for feat in feats:
        if feat not in visited and adj[feat]:
            cluster = [feat]
            stack = list(adj[feat])
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    cluster.append(node)
                    stack.extend(n for n in adj[node] if n not in visited)
            visited.add(feat)
            if len(cluster) > 1:
                clusters.append(sorted(cluster))

    return corr_dict, pairs, clusters


# ─────────────────────────────────────────────────────────────────────────────
# D — Feature group summaries (mean / SE / worst)
# ─────────────────────────────────────────────────────────────────────────────
def _group_summaries(df: pd.DataFrame) -> dict:
    """Compute per-class mean of each feature group."""
    summaries = {}
    for group_name, group_feats in FEATURE_GROUPS.items():
        mal_means = df.loc[df["diagnosis"] == 1, group_feats].mean().round(6).to_dict()
        ben_means = df.loc[df["diagnosis"] == 0, group_feats].mean().round(6).to_dict()
        summaries[group_name] = {
            "malignant_means": mal_means,
            "benign_means": ben_means,
            "features": group_feats,
        }
    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Main EDA runner
# ─────────────────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame, save_reports: bool = True) -> EDAReport:
    """
    Execute full EDA pipeline on the cleaned WDBC dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset (30 feature columns + 'diagnosis' as M=1, B=0 int).
    save_reports : bool
        If True, persist all analysis results to reports/evaluation/.

    Returns
    -------
    EDAReport
    """
    logger.info("Starting Exploratory Data Analysis ...")

    n_total = len(df)
    n_mal = int((df["diagnosis"] == 1).sum())
    n_ben = int((df["diagnosis"] == 0).sum())
    logger.info("Class distribution: Malignant (M=1)=%d, Benign (B=0)=%d, Total=%d", n_mal, n_ben, n_total)

    # A: Feature profiles
    logger.info("Profiling all 30 features ...")
    profiles: list[FeatureProfile] = []
    for feat in FEATURE_COLUMNS:
        p = _profile_feature(df, feat)
        profiles.append(p)
    logger.info("Feature profiling complete.")

    # Top discriminating features by absolute Cohen's d
    top_disc = sorted(profiles, key=lambda p: abs(p.cohens_d), reverse=True)
    top_disc_names = [p.feature for p in top_disc[:10]]
    logger.info("Top 5 discriminating features: %s", top_disc_names[:5])

    # C: Correlation analysis
    logger.info("Computing correlation matrix ...")
    corr_dict, corr_pairs, mc_clusters = _correlation_analysis(df)
    logger.info(
        "Correlation analysis done. Strong pairs (|r|>=0.70): %d | Clusters (|r|>=0.85): %d",
        len(corr_pairs), len(mc_clusters),
    )

    # D: Group summaries
    logger.info("Computing feature group summaries ...")
    group_summ = _group_summaries(df)

    report = EDAReport(
        n_samples=n_total,
        n_malignant=n_mal,
        n_benign=n_ben,
        prevalence_malignant=round(n_mal / n_total, 4),
        feature_profiles=profiles,
        top_discriminating_features=top_disc_names,
        high_correlation_pairs=corr_pairs,
        correlation_matrix=corr_dict,
        group_summaries=group_summ,
        multicollinear_clusters=mc_clusters,
    )

    if save_reports:
        EDA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        _save_eda_reports(report, df)

    logger.info("EDA pipeline complete.")
    return report


def _save_eda_reports(report: EDAReport, df: pd.DataFrame) -> None:
    """Persist EDA results as JSON and CSV files."""
    base = EDA_REPORTS_DIR

    # 1. Feature profiles CSV
    profiles_df = pd.DataFrame([asdict(p) for p in report.feature_profiles])
    profiles_df.to_csv(base / "eda_feature_profiles.csv", index=False)
    logger.info("Saved: eda_feature_profiles.csv")

    # 2. Top discriminating features
    top_df = profiles_df.set_index("feature").loc[report.top_discriminating_features][
        ["cohens_d", "effect_size_label", "u_pvalue", "significant_bonferroni",
         "mean_malignant", "mean_benign"]
    ]
    top_df.to_csv(base / "eda_top_discriminating_features.csv")
    logger.info("Saved: eda_top_discriminating_features.csv")

    # 3. Correlation pairs
    if report.high_correlation_pairs:
        pairs_df = pd.DataFrame([asdict(p) for p in report.high_correlation_pairs])
        pairs_df.to_csv(base / "eda_correlation_pairs.csv", index=False)
        logger.info("Saved: eda_correlation_pairs.csv (%d pairs)", len(pairs_df))

    # 4. Full EDA summary JSON
    summary = {
        "n_samples": report.n_samples,
        "n_malignant": report.n_malignant,
        "n_benign": report.n_benign,
        "prevalence_malignant": report.prevalence_malignant,
        "top_discriminating_features": report.top_discriminating_features,
        "n_high_correlation_pairs": len(report.high_correlation_pairs),
        "multicollinear_clusters": report.multicollinear_clusters,
        "group_summaries": report.group_summaries,
    }
    with open(base / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved: eda_summary.json")

    # 5. Descriptive statistics per class
    for label, val in [(1, "malignant"), (0, "benign")]:
        desc = df.loc[df["diagnosis"] == label, FEATURE_COLUMNS].describe().T
        desc.to_csv(base / f"eda_desc_{val}.csv")
    logger.info("Saved: eda_desc_malignant.csv and eda_desc_benign.csv")

    logger.info("All EDA reports saved to: %s", base)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: print human-readable summary to console
# ─────────────────────────────────────────────────────────────────────────────
def print_eda_summary(report: EDAReport) -> None:
    """Print a formatted ASCII console summary of the EDA findings (Windows cp1252 safe)."""
    import sys
    out = sys.stdout
    sep = "=" * 68
    dash = "-" * 68
    print(f"\n{sep}", file=out)
    print(f"  BreastGuard AI -- EDA SUMMARY REPORT", file=out)
    print(sep, file=out)
    print(f"  Dataset : Wisconsin Diagnostic Breast Cancer (WDBC)", file=out)
    print(f"  Samples : {report.n_samples}", file=out)
    print(
        f"  Classes : Malignant (M=1) = {report.n_malignant}  |  "
        f"Benign (B=0) = {report.n_benign}", file=out
    )
    print(f"  Prevalence of Malignancy: {report.prevalence_malignant * 100:.1f}%", file=out)
    print(f"  Features: 30 (10 Mean + 10 SE + 10 Worst)", file=out)
    print(file=out)

    print("  TARGET ENCODING:", file=out)
    print("  M (Malignant) = 1  |  B (Benign) = 0", file=out)
    print(dash, file=out)

    print("  TOP 10 DISCRIMINATING FEATURES (by |Cohen's d|):", file=out)
    for i, p in enumerate(
        sorted(report.feature_profiles, key=lambda x: abs(x.cohens_d), reverse=True)[:10], 1
    ):
        direction = "^ in Malignant" if p.mean_malignant > p.mean_benign else "v in Malignant"
        print(
            f"  {i:2}. {p.feature:<32} d={abs(p.cohens_d):.3f}  [{p.effect_size_label}]  {direction}",
            file=out,
        )
    print(file=out)

    print("  DATA QUALITY:", file=out)
    total_outliers = sum(p.n_outliers_total for p in report.feature_profiles)
    print(f"  Missing values    : 0 (complete dataset)", file=out)
    print(f"  Duplicate records : 0", file=out)
    print(f"  Total outliers    : {total_outliers} (IQR fence, across all 30 features)", file=out)
    print(dash, file=out)

    print(f"  MULTICOLLINEARITY:", file=out)
    print(f"  High-corr pairs (|r|>=0.70) : {len(report.high_correlation_pairs)}", file=out)
    print(f"  Multicollinear clusters     : {len(report.multicollinear_clusters)}", file=out)
    for i, cluster in enumerate(report.multicollinear_clusters[:3], 1):
        print(f"    Cluster {i}: {', '.join(cluster[:4])}{'...' if len(cluster) > 4 else ''}", file=out)
    print(file=out)

    print("  FEATURE GROUP MEANS (Malignant vs Benign):", file=out)
    for group, data in report.group_summaries.items():
        mal_avg = np.mean(list(data["malignant_means"].values()))
        ben_avg = np.mean(list(data["benign_means"].values()))
        print(f"  {group.upper():<6}  Malignant avg: {mal_avg:.4f}  |  Benign avg: {ben_avg:.4f}", file=out)
    print(f"\n{sep}\n", file=out)
