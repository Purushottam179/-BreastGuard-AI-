"""
BreastGuard AI — Page 05: Responsible AI & Limitations
Medical disclaimer, ethical considerations, model limitations, and fairness notes.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import streamlit as st

st.markdown('<h2 class="section-header">⚖️ Responsible AI & Limitations</h2>', unsafe_allow_html=True)

# ── Primary disclaimer ─────────────────────────────────────────────────────
st.markdown(
    """
    <div style="background:linear-gradient(135deg,#2D1B1B,#3D2020);
                border:2px solid #FC8181; border-radius:12px;
                padding:1.5rem 2rem; margin-bottom:1.5rem;">
        <div style="font-size:1.5rem; margin-bottom:0.5rem;">⚕️ Medical Disclaimer</div>
        <div style="color:#FED7D7; font-size:1rem; line-height:1.7;">
            <strong>BreastGuard AI is an educational research prototype.</strong><br><br>
            It is <strong>NOT a medical diagnostic device</strong> and must not be used to make
            clinical decisions. This tool was developed for academic demonstration and AI/ML
            competition purposes only.<br><br>
            <ul>
                <li>Model predictions are <strong>not confirmed diagnoses</strong>.</li>
                <li>Probability scores are <strong>not clinically validated certainty</strong>.</li>
                <li>The Wisconsin Diagnostic Breast Cancer dataset, while widely used in ML
                    research, does not represent the full diversity of real-world clinical populations.</li>
                <li><strong>Always consult a qualified medical professional</strong> for any
                    health-related questions or decisions.</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔬 Model Limitations", "⚖️ Fairness & Bias", "🧠 XAI Limitations", "📚 Dataset Caveats"]
)

with tab1:
    st.markdown("### 🔬 Known Model Limitations")
    limitations = [
        ("🎯 Accuracy ≠ Reliability",
         "High test-set accuracy on this benchmark dataset does not guarantee reliable "
         "performance in real clinical settings with different patient populations, imaging "
         "protocols, or tissue preparation methods."),
        ("🔄 Distribution Shift",
         "Models trained on data from a specific time and institution may fail to generalise "
         "to data from different hospitals, demographics, or collection methods. This is called "
         "distribution shift and is a major practical challenge in clinical AI."),
        ("📏 Small Dataset",
         "569 samples is very small for a deployed ML system. The models may overfit to "
         "idiosyncrasies of this specific dataset. Confidence intervals on all metrics are wide."),
        ("❌ False Negatives Are Costly",
         "A false negative (predicting benign when actually malignant) carries high clinical "
         "cost. This model has not been optimised with clinical cost functions. Do not use "
         "it as a screening tool."),
        ("🧬 Features Are Derived",
         "The 30 features are computed from digitised FNA images, not raw images. The "
         "quality and reliability of those feature extraction methods is assumed but not validated here."),
        ("⚙️ No Calibration",
         "Model probability outputs have not been calibrated (e.g. via Platt scaling or "
         "isotonic regression). The reported probabilities are model scores, not true "
         "frequentist probabilities."),
    ]
    for title, body in limitations:
        with st.expander(title, expanded=False):
            st.markdown(body)

with tab2:
    st.markdown("### ⚖️ Fairness & Bias Considerations")
    st.info(
        "**Known demographic gap**: The WDBC dataset does not include protected attribute "
        "information (sex assigned at birth, race, ethnicity, age, socioeconomic status). "
        "It is therefore impossible to audit this model for disparate impact across "
        "demographic subgroups."
    )
    fairness_points = {
        "Representation Bias": (
            "The dataset originates from patients at the University of Wisconsin "
            "Hospital, 1992–1995. It likely does not represent the global distribution of "
            "breast cancer presentations across different ethnicities and geographies."
        ),
        "Label Bias": (
            "The ground-truth labels (malignant/benign) were determined by clinical "
            "assessment at one institution. Label quality and consistency across other "
            "institutions is unknown."
        ),
        "Deployment Bias": (
            "Even if training data were balanced, deployment in an under-resourced "
            "clinical setting with different imaging equipment would introduce "
            "bias not captured here."
        ),
        "Model Selection Bias": (
            "The model was selected based on AUC on a held-out test set. Different "
            "demographic subgroups may experience different error rates."
        ),
    }
    for k, v in fairness_points.items():
        st.markdown(f"**{k}**")
        st.markdown(v)
        st.markdown("---")

with tab3:
    st.markdown("### 🧠 Limitations of SHAP Explanations")
    xai_limits = [
        ("Explanations Are Approximate",
         "For non-tree models (Logistic Regression, SVM), SHAP uses approximations. "
         "The KernelExplainer for SVM samples a small background dataset, making the "
         "explanation an estimate, not an exact decomposition."),
        ("Feature Independence Assumption",
         "SHAP marginalises over features assuming approximate independence. When features "
         "are highly correlated (as many WDBC features are), SHAP values can be "
         "misleading — importance may be distributed arbitrarily between correlated features."),
        ("Local ≠ Global",
         "A local SHAP explanation (for one prediction) describes that specific instance. "
         "It does not reliably explain the model's global decision boundaries."),
        ("Correlation Clusters",
         "The 30 WDBC features include mean, standard error, and worst-case values for "
         "10 measurements. These are inherently correlated. SHAP may spread importance "
         "across all three variants of a measurement rather than identifying the root "
         "causal factor."),
        ("Not Causal",
         "SHAP explanations are associative, not causal. A high SHAP value for a feature "
         "means the model relied on it — not that the feature causally drives malignancy."),
    ]
    for title, body in xai_limits:
        with st.expander(title):
            st.markdown(body)

with tab4:
    st.markdown("### 📚 Dataset Caveats")
    st.markdown("""
    **Wisconsin Diagnostic Breast Cancer (WDBC) Dataset**

    | Property | Value |
    |---|---|
    | Source | University of Wisconsin Hospital, Madison |
    | Collection period | 1992–1995 |
    | Original authors | Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian |
    | Repository | UCI ML Repository (ID 17) |
    | sklearn version | `sklearn.datasets.load_breast_cancer()` |
    | Samples | 569 |
    | Features | 30 (10 measurements × 3 statistics: mean, SE, worst) |
    | Target | Malignant (0): 212, Benign (1): 357 |
    | Missing values | None |
    | License | CC BY 4.0 (public domain for research) |

    **Citation**:
    > W.N. Street, W.H. Wolberg, O.L. Mangasarian. *Nuclear feature extraction for
    > breast tumor diagnosis.* IS&T/SPIE 1993 International Symposium on Electronic
    > Imaging: Science and Technology, volume 1905, pages 861–870, San Jose, CA, 1993.

    **Important note**: This dataset is ~30 years old. Modern breast cancer diagnosis
    uses significantly more advanced techniques including genomic profiling, MRI,
    digital pathology, and AI-enhanced imaging. The WDBC is a valuable ML benchmark
    but is not representative of contemporary clinical practice.
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4A5568; font-size:0.8rem; padding:1rem;'>"
    "BreastGuard AI · Educational Research Prototype · "
    "Built with scikit-learn, SHAP, XGBoost, Streamlit, and Plotly<br>"
    "Wisconsin Diagnostic Breast Cancer Dataset · UCI ML Repository · CC BY 4.0"
    "</div>",
    unsafe_allow_html=True,
)
