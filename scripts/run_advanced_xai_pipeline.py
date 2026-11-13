"""
BreastGuard AI — Advanced Explainability & Counterfactual Pipeline
================================================================================
Orchestrates:
  1. Interactive SHAP explorer initialization.
  2. Individual patient sample attribution and ranking.
  3. Empirical Benign vs. Malignant statistical cohort comparisons.
  4. Counterfactual sensitivity generation and minimal distance ranking.
  5. What-If single-feature trajectory exploration and tipping point discovery.
  6. Mathematical consistency and additivity audits.
  7. High-resolution figure export to reports/figures/advanced_xai/.
  8. Structured JSON artifact generation for dashboard and report integration.
================================================================================
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import pandas as pd
from config.config import EVALUATION_DIR, FIGURES_DIR
from src.explain import plot_shap_waterfall
from src.utils import setup_logger
from src.xai_explorer import (
    ADVANCED_XAI_FIGURES_DIR,
    XAIExplorer,
    plot_cohort_boxplots,
    plot_counterfactual_comparison,
    plot_explanation_consistency_audit,
    plot_feature_contribution_rankings,
    plot_what_if_trajectory,
)

logger = setup_logger("advanced_xai_pipeline")


def run_advanced_pipeline() -> dict:
    logger.info("=" * 80)
    logger.info("BreastGuard AI — Advanced Interactive Explainability Pipeline")
    logger.info("=" * 80)

    explorer = XAIExplorer()
    logger.info("XAI Explorer initialized with model: %s", explorer.model_bundle.get("model_name"))

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Individual Sample Explanation
    # ─────────────────────────────────────────────────────────────────────────
    # Find representative malignant sample (e.g. index with target=1)
    df = explorer.df_data
    t_col = explorer.target_col
    mal_idx = int(df[df[t_col] == 1].index[0])
    ben_idx = int(df[df[t_col] == 0].index[0])

    logger.info("[STEP 1] Generating individual explanation for Malignant patient (Index=%d)...", mal_idx)
    mal_exp = explorer.explain_sample_by_index(mal_idx, nsamples=120)

    # Export individual sample visualizations
    ADVANCED_XAI_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p_waterfall = plot_shap_waterfall(
        mal_exp,
        output_path=ADVANCED_XAI_FIGURES_DIR / "individual_sample_explanation.png",
    )
    p_ranking = plot_feature_contribution_rankings(
        mal_exp,
        output_path=ADVANCED_XAI_FIGURES_DIR / "feature_contribution_ranking.png",
        top_n=10,
    )
    logger.info("Saved individual sample figures to: %s", ADVANCED_XAI_FIGURES_DIR)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Benign vs. Malignant Cohort Comparison
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[STEP 2] Performing Benign vs. Malignant cohort statistical comparison...")
    top_comparison_feats = [
        "worst_texture", "worst_area", "worst_radius",
        "mean_concave_points", "worst_concavity", "worst_symmetry",
    ]
    cohort_res = explorer.compare_benign_vs_malignant(top_comparison_feats)
    cohort_csv = EVALUATION_DIR / "cohort_feature_comparison.csv"
    cohort_res.summary_table.to_csv(cohort_csv, index=False)

    p_cohort_box = plot_cohort_boxplots(
        df,
        top_comparison_feats,
        output_path=ADVANCED_XAI_FIGURES_DIR / "benign_malignant_comparison.png",
    )
    logger.info("Saved cohort comparison table to: %s", cohort_csv)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Counterfactual Explanations
    # ─────────────────────────────────────────────────────────────────────────
    # Use Patient #99 (borderline malignant, P=0.539) to illustrate sparse k=1 counterfactuals
    borderline_idx = 99
    logger.info("[STEP 3] Generating Counterfactual Explanation for Borderline Patient (Index=%d)...", borderline_idx)
    sample_borderline = df.iloc[borderline_idx][explorer.feature_names]
    cf_result = explorer.run_counterfactual_analysis(
        sample=sample_borderline,
        sample_id=borderline_idx,
        influential_features=top_comparison_feats,
    )

    logger.info("Counterfactual Found: %s", cf_result.counterfactual_found)
    if cf_result.best_counterfactual:
        bc = cf_result.best_counterfactual
        logger.info(
            "Best Counterfactual: Target=%s | P(Mal): %.4f -> %.4f | Dist: %.4f | Sparsity: %d | Strategy: %s",
            bc.predicted_label, cf_result.original_prob_malignant, bc.prob_malignant,
            bc.normalized_distance, bc.sparsity, bc.strategy,
        )
        for f, details in bc.modified_features.items():
            logger.info("  * %s: %.2f -> %.2f (%+.1f%%)", f, details["original"], details["counterfactual"], details["pct_change"])

    p_cf = plot_counterfactual_comparison(
        cf_result,
        output_path=ADVANCED_XAI_FIGURES_DIR / "counterfactual_comparison.png",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: What-If Sensitivity Analysis
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[STEP 4] Executing What-If sensitivity trajectory for 'worst_area' (Index=%d)...", borderline_idx)
    what_if_res = explorer.run_what_if_analysis(
        sample=sample_borderline,
        feature_to_sweep="worst_area",
        n_points=40,
    )

    logger.info(
        "What-If Results for 'worst_area': Original=%.2f (P=%.4f) | Tipping Point=%s | Range=[%.4f, %.4f]",
        what_if_res.original_value, what_if_res.original_prob_malignant,
        str(what_if_res.tipping_point_value), what_if_res.min_prob, what_if_res.max_prob,
    )

    p_what_if = plot_what_if_trajectory(
        what_if_res,
        output_path=ADVANCED_XAI_FIGURES_DIR / "what_if_analysis.png",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Explanation Consistency & Additivity Audit
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[STEP 5] Auditing explanation consistency and mathematical additivity...")
    audit_data = explorer.audit_explanation_consistency(test_samples=[mal_idx, ben_idx], n_repeats=3)
    logger.info("Audit Summary: All Audits Passed=%s", audit_data["all_audits_passed"])

    p_consistency = plot_explanation_consistency_audit(
        audit_data,
        output_path=ADVANCED_XAI_FIGURES_DIR / "explanation_consistency.png",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Save Comprehensive Results JSON Artifact
    # ─────────────────────────────────────────────────────────────────────────
    best_cf_dict = None
    if cf_result.best_counterfactual:
        best_cf_dict = {
            "predicted_label": cf_result.best_counterfactual.predicted_label,
            "prob_malignant": cf_result.best_counterfactual.prob_malignant,
            "prob_benign": cf_result.best_counterfactual.prob_benign,
            "modified_features": cf_result.best_counterfactual.modified_features,
            "normalized_distance": cf_result.best_counterfactual.normalized_distance,
            "sparsity": cf_result.best_counterfactual.sparsity,
            "strategy": cf_result.best_counterfactual.strategy,
        }

    advanced_summary = {
        "individual_sample_explanation": {
            "sample_index": mal_idx,
            "predicted_label": mal_exp.predicted_label,
            "prob_malignant": mal_exp.prob_malignant,
            "prob_benign": mal_exp.prob_benign,
            "threshold": mal_exp.threshold_applied,
            "base_value": mal_exp.base_value,
            "top_positive_contributions": mal_exp.top_positive_contributions,
            "top_negative_contributions": mal_exp.top_negative_contributions,
            "additivity_gap": mal_exp.additivity_gap,
            "narrative": mal_exp.narrative_explanation,
        },
        "benign_vs_malignant_comparison": {
            "features_analyzed": top_comparison_feats,
            "relative_differences_pct": cohort_res.relative_difference_pct,
        },
        "counterfactual_analysis": {
            "original_sample_id": mal_idx,
            "original_label": cf_result.original_label,
            "original_prob_malignant": cf_result.original_prob_malignant,
            "target_label": cf_result.target_label,
            "counterfactual_found": cf_result.counterfactual_found,
            "best_counterfactual": best_cf_dict,
            "safety_disclaimer": cf_result.safety_disclaimer,
        },
        "what_if_sensitivity": {
            "feature_swept": what_if_res.feature_name,
            "original_value": what_if_res.original_value,
            "original_prob": what_if_res.original_prob_malignant,
            "tipping_point_value": what_if_res.tipping_point_value,
            "probability_range": [what_if_res.min_prob, what_if_res.max_prob],
            "disclaimer": what_if_res.disclaimer,
        },
        "explanation_consistency_audit": audit_data,
        "figure_paths": {
            "individual_sample_explanation": str(p_waterfall),
            "feature_contribution_ranking": str(p_ranking),
            "benign_malignant_comparison": str(p_cohort_box),
            "counterfactual_comparison": str(p_cf),
            "what_if_analysis": str(p_what_if),
            "explanation_consistency": str(p_consistency),
        },
    }

    output_json = EVALUATION_DIR / "advanced_xai_summary.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(advanced_summary, f, indent=2)
    logger.info("Saved advanced XAI summary JSON to: %s", output_json)

    logger.info("=" * 80)
    logger.info("Advanced Explainability Pipeline Completed Successfully!")
    logger.info("=" * 80)
    return advanced_summary


if __name__ == "__main__":
    run_advanced_pipeline()
