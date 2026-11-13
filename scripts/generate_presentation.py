"""
BreastGuard AI — Competition Presentation Generator
Generates a 15-slide widescreen PowerPoint presentation (.pptx) with
professional layouts, benchmark metrics, styled tables, cards, and embedded figures.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_PPTX = PROJECT_ROOT / "reports" / "presentation" / "BreastGuard_AI_Competition_Presentation.pptx"
OUTPUT_PPTX.parent.mkdir(parents=True, exist_ok=True)

# Color Palette
CLR_DARK_BG = RGBColor(15, 23, 42)       # Slate 900
CLR_CARD_BG = RGBColor(30, 41, 59)       # Slate 800
CLR_LIGHT_BG = RGBColor(248, 250, 252)   # Slate 50
CLR_CARD_LIGHT = RGBColor(255, 255, 255) # Pure White
CLR_TEAL = RGBColor(14, 165, 233)        # Sky/Teal 500
CLR_CORAL = RGBColor(244, 63, 94)        # Rose/Coral 500
CLR_EMERALD = RGBColor(16, 185, 129)     # Emerald 500
CLR_TEXT_DARK = RGBColor(15, 23, 42)     # Text Primary Dark
CLR_TEXT_LIGHT = RGBColor(241, 245, 249) # Text Light
CLR_TEXT_MUTED = RGBColor(100, 116, 139) # Text Muted
CLR_ACCENT_BLUE = RGBColor(59, 130, 246) # Blue 500
CLR_BORDER = RGBColor(226, 232, 240)     # Slate 200

def create_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text: str, category_text: str = "BREASTGUARD AI — COMPETITION MONOGRAPH"):
        # Category / Kicker
        kicker_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.3))
        tf_k = kicker_box.text_frame
        tf_k.word_wrap = True
        p_k = tf_k.paragraphs[0]
        p_k.text = category_text.upper()
        p_k.font.size = Pt(10)
        p_k.font.bold = True
        p_k.font.color.rgb = CLR_TEAL

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.6))
        tf_t = title_box.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.color.rgb = CLR_TEXT_DARK

        # Top subtle accent line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.3), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = CLR_BORDER
        line.line.color.rgb = CLR_BORDER

    def add_card(slide, left, top, width, height, bg_color=CLR_LIGHT_BG, border_color=CLR_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
        return card

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (Dark Premium Theme)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = CLR_DARK_BG
    bg1.line.fill.background()

    # Title Card Text
    tbox = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(4.5))
    tf1 = tbox.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "RESEARCH & COMPETITION PRESENTATION"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = CLR_TEAL
    p0.space_after = Pt(14)

    p1 = tf1.add_paragraph()
    p1.text = "BreastGuard AI"
    p1.font.size = Pt(44)
    p1.font.bold = True
    p1.font.color.rgb = CLR_TEXT_LIGHT
    p1.space_after = Pt(8)

    p2 = tf1.add_paragraph()
    p2.text = "Explainable Breast Cancer Classification & Trustworthy AI Dashboard"
    p2.font.size = Pt(22)
    p2.font.color.rgb = CLR_TEAL
    p2.space_after = Pt(24)

    p3 = tf1.add_paragraph()
    p3.text = "A Rigorous, Leakage-Free Machine Learning Pipeline with Game-Theoretic SHAP Attributions, Clinical Threshold Optimization (τ = 0.37), and Out-of-Distribution Guardrails."
    p3.font.size = Pt(14)
    p3.font.color.rgb = CLR_TEXT_MUTED
    p3.space_after = Pt(36)

    p4 = tf1.add_paragraph()
    p4.text = "Author: Machine Learning & Computational Pathology Research Team | WDBC Cohort (N=569)\nNotice: Educational Research Prototype. Not an FDA/CE-cleared clinical medical device."
    p4.font.size = Pt(11)
    p4.font.color.rgb = RGBColor(148, 163, 184)

    # =========================================================================
    # SLIDE 2: THE CLINICAL DILEMMA & ASYMMETRIC LOSS
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "The Clinical Dilemma & Asymmetric Misclassification Cost")

    # Card 1: The Black-Box Problem
    add_card(s2, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.2))
    tb = s2.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(3.2), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "1. The Black-Box Trap"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = CLR_TEXT_DARK
    p.space_after = Pt(12)

    bullets = [
        "High nominal accuracy (98%+) is common in literature, but opaque decision surfaces induce clinician distrust.",
        "Clinicians cannot ethically or legally act on an uninterpretable scalar risk score without knowing underlying cytological drivers.",
        "Feature interactions and collinearity often lead models to rely on spurious non-biological artifacts."
    ]
    for b in bullets:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12)
        pb.font.color.rgb = CLR_TEXT_MUTED
        pb.space_after = Pt(8)

    # Card 2: Asymmetric Risk
    add_card(s2, Inches(4.8), Inches(1.6), Inches(3.6), Inches(5.2))
    tb = s2.shapes.add_textbox(Inches(5.0), Inches(1.8), Inches(3.2), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "2. Asymmetric Clinical Costs"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = CLR_CORAL
    p.space_after = Pt(12)

    bullets = [
        "False Negative (Type II): Carcinoma missed. Delayed treatment leads to metastasis and death. C_FN is catastrophic.",
        "False Positive (Type I): Benign case flagged. Results in confirmatory biopsy and anxiety. C_FP is moderate.",
        "Default τ = 0.50 cutoff naively assumes C_FN = C_FP, prioritizing overall accuracy over patient survival."
    ]
    for b in bullets:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12)
        pb.font.color.rgb = CLR_TEXT_MUTED
        pb.space_after = Pt(8)

    # Card 3: Our Solution
    add_card(s2, Inches(8.8), Inches(1.6), Inches(3.733), Inches(5.2))
    tb = s2.shapes.add_textbox(Inches(9.0), Inches(1.8), Inches(3.333), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "3. BreastGuard AI Solution"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = CLR_EMERALD
    p.space_after = Pt(12)

    bullets = [
        "Cost-Sensitive Threshold Tuning: Calibrated τ = 0.37 optimizes Youden's J, reducing missed cancers by 62.5%.",
        "Axiomatic Explainability: Exact SHAP game-theoretic decomposition in physical cell dimensions (μm).",
        "Trustworthy AI Guardrails: Expected Calibration Error (ECE < 2.5%) and Mahalanobis Out-of-Distribution safety."
    ]
    for b in bullets:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12)
        pb.font.color.rgb = CLR_TEXT_MUTED
        pb.space_after = Pt(8)

    # =========================================================================
    # SLIDE 3: DATASET & NUCLEAR MORPHOLOGY (WDBC Cohort)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Wisconsin Diagnostic Breast Cancer (WDBC) Dataset")

    # Left Stats Column
    add_card(s3, Inches(0.8), Inches(1.6), Inches(4.0), Inches(5.2))
    tb = s3.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(3.6), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Cohort Specifications"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = CLR_TEXT_DARK
    p.space_after = Pt(14)

    stats = [
        ("Source", "UCI ML Repository / Wolberg et al. (1995)"),
        ("Biopsies (N)", "569 fine needle aspirate (FNA) samples"),
        ("Benign (Class 0)", "357 records (62.74%)"),
        ("Malignant (Class 1)", "212 records (37.26%)"),
        ("Continuous Features", "30 morphometric measurements"),
        ("Missing / Inf Values", "0 (0.0%) — Cleaned & Validated"),
        ("ID Column", "Stripped to eliminate memorization"),
        ("Split Strategy", "80% Train (455) / 20% Test (114)")
    ]
    for k, v in stats:
        p_row = tf.add_paragraph()
        p_row.text = f"{k}: {v}"
        p_row.font.size = Pt(11)
        p_row.font.color.rgb = CLR_TEXT_MUTED
        p_row.space_after = Pt(6)

    # Right Feature Group Table
    add_card(s3, Inches(5.1), Inches(1.6), Inches(7.433), Inches(5.2))
    tb = s3.shapes.add_textbox(Inches(5.3), Inches(1.8), Inches(7.0), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Nuclear Feature Architecture (10 Attributes × 3 Aggregations = 30 Dimensions)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CLR_TEXT_DARK
    p.space_after = Pt(10)

    p_desc = tf.add_paragraph()
    p_desc.text = "Derived from digitized microscope slides via Xcyt automated contour snake models:"
    p_desc.font.size = Pt(12)
    p_desc.font.color.rgb = CLR_TEXT_MUTED
    p_desc.space_after = Pt(12)

    rows = [
        ("Mean (_mean)", "Average nuclear dimension across all segmented cells in the biopsy."),
        ("Standard Error (_se)", "Cell-to-cell variability and structural heterogeneity across the smear."),
        ("Worst (_worst)", "Mean of the 3 most extreme cells — primary pathological driver of malignancy.")
    ]
    for name, desc in rows:
        p_item = tf.add_paragraph()
        p_item.text = f"• {name}: {desc}"
        p_item.font.size = Pt(12)
        p_item.font.bold = False
        p_item.font.color.rgb = CLR_TEXT_DARK
        p_item.space_after = Pt(8)

    p_chars = tf.add_paragraph()
    p_chars.text = "Morphological Attributes:\nRadius, Texture (chromatin clumpiness), Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points (contour indentations), Symmetry, Fractal Dimension."
    p_chars.font.size = Pt(11)
    p_chars.font.color.rgb = CLR_TEXT_MUTED

    # =========================================================================
    # SLIDE 4: END-TO-END WORKFLOW & LEAKAGE PREVENTION
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "End-to-End ML Pipeline Architecture & Zero-Leakage Protocol")

    workflow_steps = [
        ("1. Data Ingestion", "UCI WDBC raw extract with automated offline fallback."),
        ("2. Schema Validation", "Sanitization, 0 NaNs, 0 Infs, M=1 / B=0 mapping."),
        ("3. Stratified 80/20 Split", "Train N=455, Test N=114 isolated before modeling."),
        ("4. Pipeline Scaler", "StandardScaler fitted strictly inside training folds."),
        ("5. 5-Fold Stratified CV", "Balanced class weighting across 5 distinct folds."),
        ("6. Hyperparameter Search", "Grid and Randomized search across 5 model families."),
        ("7. Champion Selection", "RBF-SVM chosen for highest AUC and lowest variance."),
        ("8. Threshold Tuning", "Youden's J optimizes cutoff τ = 0.37 on OOF probs."),
        ("9. Held-Out Evaluation", "114 test samples unlocked for final unbiased testing."),
        ("10. Explainable AI", "KernelExplainer calculates exact unscaled SHAP values."),
        ("11. Responsible AI", "ECE calibration, Mahalanobis OOD, noise tests."),
        ("12. Clinical Dashboard", "7-page Streamlit interface + vector PDF reports.")
    ]

    for idx, (step_title, step_desc) in enumerate(workflow_steps):
        col = idx // 4
        row = idx % 4
        left = Inches(0.8 + col * 4.0)
        top = Inches(1.6 + row * 1.35)
        add_card(s4, left, top, Inches(3.7), Inches(1.2))

        tb = s4.shapes.add_textbox(left + Inches(0.15), top + Inches(0.1), Inches(3.4), Inches(1.0))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = step_title
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = CLR_ACCENT_BLUE
        p.space_after = Pt(2)

        pd = tf.add_paragraph()
        pd.text = step_desc
        pd.font.size = Pt(10)
        pd.font.color.rgb = CLR_TEXT_MUTED

    # =========================================================================
    # SLIDE 5: MODEL BENCHMARKING (5 Algorithms Compared)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Multi-Model Cross-Validation Benchmarking (N=455 Training Split)")

    # Left: Summary Table
    add_card(s5, Inches(0.8), Inches(1.6), Inches(7.5), Inches(5.2))
    tb = s5.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(7.1), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "5-Fold Stratified Cross-Validation Benchmark Scorecard"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CLR_TEXT_DARK
    p.space_after = Pt(10)

    models_data = [
        ("Support Vector Machine (Champion)", "0.9780 ± 0.007", "0.9588 ± 0.030", "0.9701 ± 0.010", "0.9959 ± 0.005", "0.0179"),
        ("Logistic Regression (L1 Tuned)", "0.9736 ± 0.011", "0.9471 ± 0.034", "0.9638 ± 0.016", "0.9959 ± 0.004", "0.0199"),
        ("Gradient Boosting", "0.9736 ± 0.020", "0.9471 ± 0.034", "0.9639 ± 0.028", "0.9934 ± 0.006", "0.0220"),
        ("K-Nearest Neighbors (k=11)", "0.9604 ± 0.015", "0.9059 ± 0.043", "0.9443 ± 0.022", "0.9892 ± 0.011", "0.0298"),
        ("Random Forest (100 Trees)", "0.9560 ± 0.007", "0.9412 ± 0.026", "0.9411 ± 0.010", "0.9899 ± 0.005", "0.0336")
    ]

    p_tbl = tf.add_paragraph()
    p_tbl.text = "Model | Accuracy | Malignant Recall | F1-Score | ROC-AUC | Brier Score\n" + "-"*75
    p_tbl.font.size = Pt(10)
    p_tbl.font.bold = True
    p_tbl.font.color.rgb = CLR_TEXT_DARK

    for m, acc, rec, f1, auc, br in models_data:
        p_row = tf.add_paragraph()
        p_row.text = f"{m}\n  Acc: {acc} | Recall: {rec} | F1: {f1} | AUC: {auc} | Brier: {br}"
        p_row.font.size = Pt(10)
        p_row.font.color.rgb = CLR_ACCENT_BLUE if "Champion" in m else CLR_TEXT_MUTED
        p_row.space_after = Pt(4)

    # Right: Why SVM Won
    add_card(s5, Inches(8.6), Inches(1.6), Inches(3.933), Inches(5.2))
    tb_r = s5.shapes.add_textbox(Inches(8.8), Inches(1.8), Inches(3.533), Inches(4.8))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p_rw = tf_r.paragraphs[0]
    p_rw.text = "Why Support Vector Machine Won"
    p_rw.font.size = Pt(16)
    p_rw.font.bold = True
    p_rw.font.color.rgb = CLR_TEXT_DARK
    p_rw.space_after = Pt(12)

    reasons = [
        ("Highest ROC-AUC (0.9959)", "Superior boundary separation in infinite-dimensional RKHS."),
        ("Lowest Fold Variance (σ=0.0054)", "Exceptionally stable across folds without tree overfitting."),
        ("Top Malignant Sensitivity (95.88%)", "Class-weighted margin directly penalizes missed carcinomas."),
        ("Collinearity Immunity", "L2 regularized margin handles correlated cell dimensions gracefully."),
        ("Best Probability Calibration", "Platt scaling achieved top CV Brier score (0.0179).")
    ]
    for title, desc in reasons:
        p_it = tf_r.add_paragraph()
        p_it.text = f"• {title}: {desc}"
        p_it.font.size = Pt(11)
        p_it.font.color.rgb = CLR_TEXT_MUTED
        p_it.space_after = Pt(8)

    # =========================================================================
    # SLIDE 6: ASYMMETRIC THRESHOLD CALIBRATION (τ = 0.37)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Clinical Threshold Optimization via Youden's J Statistic")

    # Left: Methodology
    add_card(s6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    tb = s6.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "The Mathematical Cutoff Shift"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = CLR_TEXT_DARK
    p.space_after = Pt(12)

    p_eq = tf.add_paragraph()
    p_eq.text = "Youden's Index Optimization:\nJ(τ) = Sensitivity(τ) + Specificity(τ) - 1\nOptimal Cutoff: τ* = 0.37  (J = 0.9578)"
    p_eq.font.size = Pt(13)
    p_eq.font.bold = True
    p_eq.font.color.rgb = CLR_TEAL
    p_eq.space_after = Pt(16)

    bullets = [
        "Default 0.50 cutoff produced 8 False Negatives on out-of-fold training data.",
        "Calibrating to τ = 0.37 reduced False Negatives from 8 down to 3 (a 62.5% reduction in missed cancers).",
        "Out-of-fold sensitivity increased from 95.29% to 98.24%.",
        "Specificity remained exceptionally high at 97.54% (only 7 false alarms out of 285 benign cases)."
    ]
    for b in bullets:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(11)
        pb.font.color.rgb = CLR_TEXT_MUTED
        pb.space_after = Pt(8)

    # Right: Embed Threshold Chart
    add_card(s6, Inches(6.7), Inches(1.6), Inches(5.833), Inches(5.2))
    thresh_img = FIGURES_DIR / "threshold_clinical_error_tradeoff.png"
    if thresh_img.exists():
        s6.shapes.add_picture(str(thresh_img), Inches(6.9), Inches(1.8), width=Inches(5.433))
    else:
        tb_alt = s6.shapes.add_textbox(Inches(6.9), Inches(1.8), Inches(5.433), Inches(4.8))
        tb_alt.text_frame.text = "Threshold Tradeoff Chart: Demonstrates steep drop in False Negatives at τ=0.37."

    # =========================================================================
    # SLIDE 7: FINAL HELD-OUT TEST PERFORMANCE (N=114)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Final Evaluation on Isolated Held-Out Test Set (N=114)")

    # Left: 4 Metric Cards
    metric_cards = [
        ("99.12%", "Overall Test Accuracy", "113 / 114 Biopsies Correct", CLR_TEAL),
        ("97.62%", "Malignant Sensitivity", "41 / 42 Carcinomas Detected", CLR_CORAL),
        ("100.0%", "Benign Specificity", "72 / 72 Benign Correct (0 FP)", CLR_EMERALD),
        ("0.0157", "Test Brier Score", "Rank #1 Probability Calibration", CLR_ACCENT_BLUE)
    ]
    for idx, (val, title, sub, col_c) in enumerate(metric_cards):
        r = idx // 2
        c = idx % 2
        left = Inches(0.8 + c * 2.8)
        top = Inches(1.6 + r * 2.6)
        add_card(s7, left, top, Inches(2.6), Inches(2.4))

        tb = s7.shapes.add_textbox(left + Inches(0.1), top + Inches(0.2), Inches(2.4), Inches(2.0))
        tf = tb.text_frame
        tf.word_wrap = True
        p_v = tf.paragraphs[0]
        p_v.text = val
        p_v.font.size = Pt(28)
        p_v.font.bold = True
        p_v.font.color.rgb = col_c
        p_v.space_after = Pt(4)

        p_t = tf.add_paragraph()
        p_t.text = title
        p_t.font.size = Pt(12)
        p_t.font.bold = True
        p_t.font.color.rgb = CLR_TEXT_DARK
        p_t.space_after = Pt(2)

        p_s = tf.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(10)
        p_s.font.color.rgb = CLR_TEXT_MUTED

    # Right: Embed Confusion Matrix Image
    add_card(s7, Inches(6.7), Inches(1.6), Inches(5.833), Inches(5.2))
    cm_img = FIGURES_DIR / "confusion_matrix_test.png"
    if cm_img.exists():
        s7.shapes.add_picture(str(cm_img), Inches(7.0), Inches(1.8), width=Inches(5.2))
    else:
        tb_cm = s7.shapes.add_textbox(Inches(7.0), Inches(2.5), Inches(5.2), Inches(3.0))
        tb_cm.text_frame.text = "Held-Out Test Confusion Matrix:\nTN=72, FP=0, FN=1, TP=41"

    # =========================================================================
    # SLIDE 8: EXPLAINABLE AI: GLOBAL SHAP ATTRIBUTIONS
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Explainable AI: Global SHAP Cohort Importance")

    # Left: Ranking & Insights
    add_card(s8, Inches(0.8), Inches(1.6), Inches(5.5), Inches(5.2))
    tb = s8.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.1), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Biomarker Importance Ranking"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = CLR_TEXT_DARK
    p.space_after = Pt(10)

    top_feats = [
        ("1. worst_texture", "0.0368", "Chromatin clumping & nuclear hyperchromasia"),
        ("2. worst_symmetry", "0.0325", "Nuclear contour shape distortion & asymmetry"),
        ("3. worst_radius", "0.0278", "Pronounced enlargement in largest cell nuclei"),
        ("4. worst_area", "0.0260", "Volumetric expansion of dysplastic cells"),
        ("5. mean_concave_pts", "0.0259", "Frequency of membrane perimeter indentations"),
        ("6. worst_concavity", "0.0256", "Depth and severity of structural invaginations")
    ]
    for feat, val, desc in top_feats:
        pf = tf.add_paragraph()
        pf.text = f"{feat} (|SHAP|={val}): {desc}"
        pf.font.size = Pt(11)
        pf.font.color.rgb = CLR_TEXT_DARK
        pf.space_after = Pt(4)

    p_conc = tf.add_paragraph()
    p_conc.text = "\nClinical Finding: The top 4 predictors are all 'Worst' dimensions. The model accurately mirrors cytopathology by focusing on the most dysplastic cells rather than bulk averages."
    p_conc.font.size = Pt(11)
    p_conc.font.bold = True
    p_conc.font.color.rgb = CLR_ACCENT_BLUE

    # Right: Embed Beeswarm Image
    add_card(s8, Inches(6.6), Inches(1.6), Inches(5.933), Inches(5.2))
    beeswarm_img = FIGURES_DIR / "shap" / "shap_summary_beeswarm.png"
    if beeswarm_img.exists():
        s8.shapes.add_picture(str(beeswarm_img), Inches(6.8), Inches(1.8), width=Inches(5.533))

    # =========================================================================
    # SLIDE 9: LOCAL WATERFALL EXPLANATIONS & RECOURSE
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Patient-Level Local Explanations & Algorithmic Recourse")

    # Left: Waterfall Explanation
    add_card(s9, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    wf_img = FIGURES_DIR / "shap" / "shap_waterfall_example.png"
    if wf_img.exists():
        s9.shapes.add_picture(str(wf_img), Inches(1.0), Inches(1.8), width=Inches(5.2))
    else:
        tb_wf = s9.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(5.2), Inches(3.0))
        tb_wf.text_frame.text = "Local Waterfall Chart: E[f(X)]=24.3% + Signed Feature Contributions."

    # Right: Axiomatic Guarantees & Recourse
    add_card(s9, Inches(6.7), Inches(1.6), Inches(5.833), Inches(5.2))
    tb_r = s9.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "Axiomatic Guarantees & Recourse"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = CLR_TEXT_DARK
    p.space_after = Pt(12)

    bullets = [
        "Exact Additivity: Sum of SHAP attributions equals f(x) - E[f(X)] with zero gap (Δ = 0.000000).",
        "Physical Units: Scaler pipeline wrapper allows attribution directly in unscaled biological units (μm, μm²).",
        "What-If Sweeps: Dynamically simulate individual feature variation across empirical percentiles (10th to 90th).",
        "Counterfactual Recourse: Computes minimal feature shift δ* required to flip borderline malignant cases into benign.",
        "Clinical Disclaimer: Counterfactuals are mathematical probes of the decision surface, not feasible biological interventions."
    ]
    for b in bullets:
        pb = tf_r.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(11)
        pb.font.color.rgb = CLR_TEXT_MUTED
        pb.space_after = Pt(8)

    # =========================================================================
    # SLIDE 10: TRUSTWORTHY AI: CALIBRATION & RELIABILITY
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Trustworthy AI: Probabilistic Calibration & Reliability")

    # Left: Embed Calibration Curves
    add_card(s10, Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.2))
    cal_img = FIGURES_DIR / "calibration_curves.png"
    if cal_img.exists():
        s10.shapes.add_picture(str(cal_img), Inches(1.0), Inches(1.8), width=Inches(5.4))

    # Right: Calibration Analysis
    add_card(s10, Inches(6.9), Inches(1.6), Inches(5.633), Inches(5.2))
    tb = s10.shapes.add_textbox(Inches(7.1), Inches(1.8), Inches(5.2), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Calibration & Certainty Analysis"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = CLR_TEXT_DARK
    p.space_after = Pt(12)

    cal_facts = [
        ("Expected Calibration Error (ECE)", "0.0242 (2.42% across 10 equal bins)"),
        ("Test Brier Score", "0.0157 (#1 among all benchmarked models)"),
        ("Random Forest ECE", "0.0520 (under-confident in extreme probabilities)"),
        ("High-Confidence Predictions", "91.2% of cohort exhibit max(P, 1-P) ≥ 0.85"),
        ("Borderline Triage Window", "3.5% of cases fall in [0.30, 0.45] uncertainty zone"),
        ("Safety Rule", "Predicted probability is distinguished from clinical certainty.")
    ]
    for k, v in cal_facts:
        pk = tf.add_paragraph()
        pk.text = f"• {k}: {v}"
        pk.font.size = Pt(11)
        pk.font.color.rgb = CLR_TEXT_MUTED
        pk.space_after = Pt(8)

    # =========================================================================
    # SLIDE 11: DISTRIBUTIONAL GUARDRAILS & OOD DETECTION
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Out-of-Distribution (OOD) Guardrails & Atypicality Alerts")

    # 3 Strategy Cards
    add_card(s11, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.2))
    tb = s11.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(3.2), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "1. Regularized Mahalanobis"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CLR_TEXT_DARK
    p.space_after = Pt(10)

    bullets = [
        "Cytological features exhibit extreme collinearity (r > 0.95). Standard covariance is ill-conditioned.",
        "We employ Ledoit-Wolf shrinkage to regularize covariance inversion.",
        "Thresholds fitted strictly on training split (N=455) to prevent leakage."
    ]
    for b in bullets:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(11)
        pb.font.color.rgb = CLR_TEXT_MUTED
        pb.space_after = Pt(8)

    add_card(s11, Inches(4.8), Inches(1.6), Inches(3.6), Inches(5.2))
    tb = s11.shapes.add_textbox(Inches(5.0), Inches(1.8), Inches(3.2), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "2. Two-Tier Thresholds"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CLR_CORAL
    p.space_after = Pt(10)

    bullets = [
        "τ_95 = 7.11 (95th percentile):\nTriggers MILD_OUTLIER alert. Flags unusual morphology for review.",
        "τ_99 = 12.75 (99th percentile):\nTriggers SEVERE_OOD alert. Rejects corrupted or non-FNA data.",
        "Dual verification via Isolation Forest anomaly depth scoring."
    ]
    for b in bullets:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(11)
        pb.font.color.rgb = CLR_TEXT_MUTED
        pb.space_after = Pt(8)

    add_card(s11, Inches(8.8), Inches(1.6), Inches(3.733), Inches(5.2))
    tb = s11.shapes.add_textbox(Inches(9.0), Inches(1.8), Inches(3.333), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "3. Clinical Safety Policy"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CLR_EMERALD
    p.space_after = Pt(10)

    bullets = [
        "Never Silent Failure: Models must acknowledge when input data falls outside the training manifold.",
        "Non-Diagnostic: An OOD alert indicates mathematical divergence, never an automated disease call.",
        "Mandatory Pathology Review: Outlier samples require slide re-staining and repeat aspiration."
    ]
    for b in bullets:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(11)
        pb.font.color.rgb = CLR_TEXT_MUTED
        pb.space_after = Pt(8)

    # =========================================================================
    # SLIDE 12: SUBGROUP EQUITY & PERTURBATION ROBUSTNESS
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Subgroup Fairness Audit & Monte Carlo Perturbation Robustness")

    # Left: Subgroup Audit
    add_card(s12, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    tb = s12.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Tumor Size Subgroup Analysis"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = CLR_TEXT_DARK
    p.space_after = Pt(12)

    strata = [
        ("Small Tumors (≤ 13.0 mm)", "100.0% Sensitivity (0 / 11 missed)", CLR_EMERALD),
        ("Medium Tumors (13.0 - 17.5 mm)", "95.24% Sensitivity (1 / 21 missed)", CLR_CORAL),
        ("Large Tumors (> 17.5 mm)", "100.0% Sensitivity (0 / 10 missed)", CLR_EMERALD)
    ]
    for s_name, s_res, col_s in strata:
        ps = tf.add_paragraph()
        ps.text = f"• {s_name}:\n  {s_res}"
        ps.font.size = Pt(12)
        ps.font.bold = True
        ps.font.color.rgb = col_s
        ps.space_after = Pt(8)

    p_dem = tf.add_paragraph()
    p_dem.text = "\nDemographic Disclosure: WDBC contains zero demographic attributes (no age, race, ethnicity). We explicitly state this limitation, refusing to make ungrounded demographic fairness claims."
    p_dem.font.size = Pt(11)
    p_dem.font.color.rgb = CLR_TEXT_MUTED

    # Right: Robustness
    add_card(s12, Inches(6.7), Inches(1.6), Inches(5.833), Inches(5.2))
    tb_r = s12.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "Monte Carlo Noise Robustness"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = CLR_TEXT_DARK
    p.space_after = Pt(12)

    bullets = [
        "Simulated 100 stochastic trials under Gaussian measurement noise (ε = 1%, 5%, 10%).",
        "Clear-Cut Cases: Maintained 100.0% classification stability at 5% measurement noise.",
        "Borderline Cases (near τ = 0.37): Exhibited expected sensitivity (76% stability), appropriately triggering caution alerts.",
        "Elasticity Ranking: Ranks features most sensitive to noise (worst_texture, worst_radius, worst_area)."
    ]
    for b in bullets:
        pb = tf_r.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(11)
        pb.font.color.rgb = CLR_TEXT_MUTED
        pb.space_after = Pt(8)

    # =========================================================================
    # SLIDE 13: INTERACTIVE DASHBOARD & REPORTING
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Interactive Streamlit Dashboard & Clinical Reporting")

    # 4 Feature Quadrants
    features = [
        ("Executive Overview & EDA", "Interactive KPI indicators, cohort balance donut, 5-model CV comparison, and 30x30 multicollinearity heatmap."),
        ("Clinical Prediction Console", "30 continuous cytological sliders grouped by Mean/SE/Worst, clinical presets, and real-time risk gauge."),
        ("Explainable AI Explorer", "Global beeswarm, patient waterfall attributions, live What-If sensitivity curves, and counterfactual recourse."),
        ("Clinical PDF Report Generator", "Downloadable publication-grade vector PDF clinical summary sheet (53 KB) with disclaimers and SHAP bars.")
    ]
    for idx, (ftitle, fdesc) in enumerate(features):
        r = idx // 2
        c = idx % 2
        left = Inches(0.8 + c * 5.9)
        top = Inches(1.6 + r * 2.6)
        add_card(s13, left, top, Inches(5.7), Inches(2.4))

        tb = s13.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), Inches(5.3), Inches(2.0))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = ftitle
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = CLR_ACCENT_BLUE
        p.space_after = Pt(8)

        pd = tf.add_paragraph()
        pd.text = fdesc
        pd.font.size = Pt(12)
        pd.font.color.rgb = CLR_TEXT_MUTED

    # =========================================================================
    # SLIDE 14: LIMITATIONS, GOVERNANCE & FUTURE ROADMAP
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Limitations, Regulatory Alignment & Future Roadmap")

    # Left: Limitations
    add_card(s14, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    tb = s14.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Documented Limitations"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = CLR_CORAL
    p.space_after = Pt(12)

    limits = [
        "Sample Size (N=569): Modest single-center historical cohort (1995).",
        "No Demographic PII: Cannot confirm equity across race, age, or ethnicity.",
        "Surrogate Explanations: SHAP explains model decisions, not biological causality.",
        "Non-Clinical Prototype: Has not undergone prospective clinical trials (FDA/CE)."
    ]
    for l in limits:
        pl = tf.add_paragraph()
        pl.text = "• " + l
        pl.font.size = Pt(11)
        pl.font.color.rgb = CLR_TEXT_MUTED
        pl.space_after = Pt(8)

    # Right: Future Scope & Governance
    add_card(s14, Inches(6.7), Inches(1.6), Inches(5.833), Inches(5.2))
    tb_r = s14.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    p = tf_r.paragraphs[0]
    p.text = "Governance & Future Roadmap"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = CLR_EMERALD
    p.space_after = Pt(12)

    future = [
        "Multi-Modal WSI Fusion: Ingest raw whole-slide microscopy images using Vision Transformers (ViT) alongside tabular features.",
        "Conformal Prediction: Provide mathematical uncertainty bounds with 99% statistical coverage guarantee.",
        "Regulatory Framework Alignment: Designed following EU AI Act Class IIb High-Risk AI and FDA Good Machine Learning Practice (GMLP).",
        "Hospital EHR Connectors: HL7 / FHIR interfaces for direct pathology laboratory integration."
    ]
    for f in future:
        pf = tf_r.add_paragraph()
        pf.text = "• " + f
        pf.font.size = Pt(11)
        pf.font.color.rgb = CLR_TEXT_MUTED
        pf.space_after = Pt(8)

    # =========================================================================
    # SLIDE 15: CONCLUSION & DEFENSE SUMMARY (Dark Theme)
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    bg15 = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg15.fill.solid()
    bg15.fill.fore_color.rgb = CLR_DARK_BG
    bg15.line.fill.background()

    tb15 = s15.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.333), Inches(5.0))
    tf15 = tb15.text_frame
    tf15.word_wrap = True

    p0 = tf15.paragraphs[0]
    p0.text = "SUMMARY & JUDGES' DEFENSE"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = CLR_TEAL
    p0.space_after = Pt(12)

    p1 = tf15.add_paragraph()
    p1.text = "Why BreastGuard AI Stands Out"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = CLR_TEXT_LIGHT
    p1.space_after = Pt(20)

    takeaways = [
        "Zero Data Leakage: 100% rigorous pipeline boundaries with isolated test sets.",
        "Clinical Asymmetry Addressed: Calibrated threshold τ = 0.37 cut missed malignancies by 62.5%.",
        "Superior Empirical Performance: 99.12% accuracy, 97.62% sensitivity, 100% specificity on held-out test data.",
        "Game-Theoretic Transparency: Unscaled SHAP attributions with verified zero additivity gap (Δ = 0.000000).",
        "Proactive Governance: Brier score 0.0157, ECE 0.0242, Ledoit-Wolf OOD guardrails, 112 passing automated tests."
    ]
    for t in takeaways:
        pt = tf15.add_paragraph()
        pt.text = "✔ " + t
        pt.font.size = Pt(14)
        pt.font.color.rgb = CLR_TEXT_LIGHT
        pt.space_after = Pt(8)

    p_qa = tf15.add_paragraph()
    p_qa.text = "\nThank you. We welcome questions from the evaluation panel."
    p_qa.font.size = Pt(16)
    p_qa.font.bold = True
    p_qa.font.color.rgb = CLR_TEAL

    prs.save(str(OUTPUT_PPTX))
    print(f"Presentation saved successfully to: {OUTPUT_PPTX}")

if __name__ == "__main__":
    create_deck()

