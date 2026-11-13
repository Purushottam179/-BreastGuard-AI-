"""
BreastGuard AI — Dashboard UI & Component Unit Tests
================================================================================
Validates that:
  1. All Streamlit page and component modules import without error.
  2. Data loaders for dashboard pages return correctly formatted dataframes and dictionaries.
  3. Interactive Plotly visualization components return valid go.Figure objects.
  4. Global navigation mapping contains all required views.
================================================================================
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from app.components import dashboard, explainability, prediction, sidebar, visualizations
from app.components.sidebar import PAGES
from app.components.visualizations import (
    render_class_distribution_donut,
    render_clinical_confusion_matrix,
    render_correlation_heatmap,
    render_cv_model_comparison_chart,
    render_feature_distribution_chart,
    render_threshold_tradeoff_chart,
)
from app.pages import (
    about,
    dataset_analysis,
    explainable_ai,
    model_performance,
    overview,
    prediction_page,
    responsible_ai,
)


class TestDashboardModuleImports:
    """Ensure all UI modules and pages are valid Python modules."""

    def test_pages_import_successfully(self) -> None:
        """Verify all page modules possess render_page callable."""
        for mod in [about, dataset_analysis, explainable_ai, model_performance, overview, prediction_page, responsible_ai]:
            assert hasattr(mod, "render_page"), f"Module {mod} missing render_page()"
            assert callable(getattr(mod, "render_page"))

    def test_sidebar_pages_registry(self) -> None:
        """Verify sidebar navigation contains all required views."""
        expected_keys = [
            "📊 Executive Overview",
            "🔬 Dataset Analysis",
            "🤖 Model Performance",
            "🔮 Breast Cancer Prediction",
            "🧠 Explainable AI (SHAP)",
            "🛡️ Responsible AI & Reliability",
            "⚖️ About Project",
        ]
        for key in expected_keys:
            assert key in PAGES, f"Sidebar missing expected page key: {key}"


class TestDashboardDataLoaders:
    """Test data loading functions used by the Streamlit pages."""

    def test_overview_data_loader(self) -> None:
        """Verify overview data loader returns valid evaluation dictionary and CV dataframe."""
        eval_data, cv_df = overview.load_overview_data()
        assert isinstance(eval_data, dict)
        assert isinstance(cv_df, pd.DataFrame)
        assert not cv_df.empty, "Cross-validation dataframe should not be empty"
        assert "Model" in cv_df.columns

    def test_dataset_analysis_loader(self) -> None:
        """Verify dataset analysis loader returns 569 cleaned biopsy records."""
        df = dataset_analysis.load_cleaned_dataset()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 569, f"Expected 569 samples, got {len(df)}"
        target_col = "diagnosis" if "diagnosis" in df.columns else "target"
        assert target_col in df.columns
        assert df[target_col].isin([0, 1]).all()

    def test_performance_artifacts_loader(self) -> None:
        """Verify model performance loader returns all required evaluation tables."""
        eval_data, cv_df, hyper_df, sweep_df = model_performance.load_performance_artifacts()
        assert isinstance(eval_data, dict)
        assert "champion_model" in eval_data
        assert eval_data["champion_model"] == "Support Vector Machine"
        assert not cv_df.empty
        assert not sweep_df.empty
        assert "sensitivity" in sweep_df.columns
        assert "specificity" in sweep_df.columns

    def test_xai_artifacts_loader(self) -> None:
        """Verify explainability artifacts loader returns SHAP summaries."""
        shap_data, adv_data = explainable_ai.load_xai_artifacts()
        assert isinstance(shap_data, dict)
        assert isinstance(adv_data, dict)
        assert "global_importance_ranking" in shap_data
        assert len(shap_data["global_importance_ranking"]) >= 10
        assert "counterfactual_analysis" in adv_data


class TestPlotlyVisualizationGenerators:
    """Verify that all visualization functions generate valid Plotly figures."""

    def test_confusion_matrix_figure(self) -> None:
        """Test clinical confusion matrix generator."""
        fig = render_clinical_confusion_matrix(tp=41, fn=1, fp=0, tn=72)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == "heatmap"

    def test_class_distribution_donut(self) -> None:
        """Test class distribution donut chart generator."""
        fig = render_class_distribution_donut(n_benign=357, n_malignant=212)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == "pie"
        assert fig.data[0].hole == 0.62

    def test_cv_model_comparison_chart(self) -> None:
        """Test CV model comparison bar chart generator."""
        _, cv_df = overview.load_overview_data()
        fig = render_cv_model_comparison_chart(cv_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # Accuracy, Recall, ROC-AUC
        assert all(t.type == "bar" for t in fig.data)

    def test_feature_distribution_chart(self) -> None:
        """Test single feature histogram and box distribution generator."""
        df = dataset_analysis.load_cleaned_dataset()
        df["Diagnosis"] = df["diagnosis"].map({0: "Benign", 1: "Malignant"})
        fig = render_feature_distribution_chart(df, "mean_radius")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Benign + Malignant traces

    def test_correlation_heatmap(self) -> None:
        """Test correlation heatmap generator."""
        df = dataset_analysis.load_cleaned_dataset()
        numeric_cols = [c for c in df.columns if c not in ["diagnosis", "target", "id"]][:10]
        corr_matrix = df[numeric_cols].corr()
        fig = render_correlation_heatmap(corr_matrix)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == "heatmap"

    def test_threshold_tradeoff_chart(self) -> None:
        """Test threshold sweep trade-off chart generator."""
        _, _, _, sweep_df = model_performance.load_performance_artifacts()
        fig = render_threshold_tradeoff_chart(sweep_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # Sensitivity, Specificity, F1
        assert any(t.name == "Sensitivity (Recall)" for t in fig.data)

