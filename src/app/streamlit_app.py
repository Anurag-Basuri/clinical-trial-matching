import sys
from pathlib import Path

# Add project root to path so 'src' imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pickle
import pandas as pd

# Project imports
from src.utils.json_loader import load_all_data
from src.privacy.anonymizer import anonymize
from src.preprocessing import preprocess


# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Clinical Trial Matcher",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .patient-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
    }
    .trial-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #43A047;
    }
    .eligible-badge {
        background: #43A047;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .not-eligible-badge {
        background: #E53935;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# =============================
# LOAD MODELS & DATA
# =============================
@st.cache_resource
def load_models():
    models_dir = PROJECT_ROOT / "models"
    try:
        with open(models_dir / "tfidf.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open(models_dir / "classifier.pkl", "rb") as f:
            classifier = pickle.load(f)
        return vectorizer, classifier, None
    except FileNotFoundError as e:
        return None, None, str(e)


@st.cache_data
def load_data():
    patients, trials, pairs = load_all_data(base_dir=PROJECT_ROOT / "data")
    return patients, trials, pairs


# =============================
# HELPER FUNCTIONS
# =============================
def get_eligibility_details(patient, trial):
    """Return detailed eligibility breakdown."""
    age = patient["metadata"]["age"]
    conds = set(patient["metadata"]["conditions"])
    c = trial["criteria"]
    
    checks = []
    all_pass = True
    
    # Age check
    age_pass = c["min_age"] <= age <= c["max_age"]
    checks.append({
        "criterion": "Age Range",
        "requirement": f"{c['min_age']} - {c['max_age']} years",
        "patient_value": f"{age} years",
        "status": "✅ Pass" if age_pass else "❌ Fail"
    })
    if not age_pass:
        all_pass = False
    
    # Required conditions
    required = set(c["required_conditions"])
    has_required = required.issubset(conds)
    missing = required - conds
    checks.append({
        "criterion": "Required Conditions",
        "requirement": ", ".join(required) if required else "None",
        "patient_value": ", ".join(required & conds) if required & conds else "None present",
        "status": "✅ Pass" if has_required else f"❌ Missing: {', '.join(missing)}"
    })
    if not has_required:
        all_pass = False
    
    # Excluded conditions
    excluded = set(c["excluded_conditions"])
    has_excluded = excluded & conds
    checks.append({
        "criterion": "Excluded Conditions",
        "requirement": f"Must NOT have: {', '.join(excluded)}" if excluded else "None",
        "patient_value": ", ".join(has_excluded) if has_excluded else "None present",
        "status": "❌ Has excluded" if has_excluded else "✅ Pass"
    })
    if has_excluded:
        all_pass = False
    
    return checks, all_pass


def get_patient_summary(patient):
    """Generate a formatted patient summary."""
    meta = patient["metadata"]
    return {
        "ID": patient["patient_id"],
        "Age": meta["age"],
        "Gender": meta["gender"].capitalize(),
        "Conditions": ", ".join(meta["conditions"]) if meta["conditions"] else "None",
        "Negated": ", ".join(meta["negated_conditions"]) if meta["negated_conditions"] else "None"
    }


def get_trial_summary(trial):
    """Generate a formatted trial summary."""
    c = trial["criteria"]
    return {
        "ID": trial["trial_id"],
        "Age Range": f"{c['min_age']} - {c['max_age']}",
        "Required": ", ".join(c["required_conditions"]) if c["required_conditions"] else "None",
        "Excluded": ", ".join(c["excluded_conditions"]) if c["excluded_conditions"] else "None"
    }


# =============================
# MAIN APP
# =============================
def main():
    # Load data and models
    vectorizer, classifier, model_error = load_models()
    patients, trials, pairs = load_data()
    ui_only_mode = model_error is not None
    
    patient_map = {p["patient_id"]: p for p in patients}
    trial_map = {t["trial_id"]: t for t in trials}
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
        st.markdown("## 🧪 Clinical Trial Matcher")
        st.markdown("---")
        
        st.markdown("### 📊 Dataset Stats")
        col1, col2 = st.columns(2)
        col1.metric("Patients", len(patients))
        col2.metric("Trials", len(trials))
        st.metric("Total Pairs", len(pairs))
        
        # Calculate eligibility rate
        if pairs:
            eligible_count = sum(1 for p in pairs if p.get("label") == 1)
            rate = eligible_count / len(pairs) * 100
            st.metric("Eligibility Rate", f"{rate:.1f}%")
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        show_raw_text = st.checkbox("Show raw text", value=True)
        if ui_only_mode:
            st.checkbox(
                "Use ML prediction",
                value=False,
                disabled=True,
                help="ML models not available for UI-only deploy."
            )
            show_model_prediction = False
        else:
            show_model_prediction = st.checkbox("Use ML prediction", value=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This app matches patients to clinical trials using:
        - **TF-IDF** text vectorization
        - **ML classifier** for prediction
        - **Rule-based** explanations
        """)
    
    # Main content
    st.markdown('<p class="main-header">🧪 Clinical Trial Eligibility Matching</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Privacy-preserving NLP-based eligibility prediction with explainable AI</p>', unsafe_allow_html=True)
    
    # Check for model errors
    if model_error:
        st.warning(f"⚠️ Model not found: {model_error}")
        st.info("UI-only mode enabled. Train models later with `python run_pipeline.py`.")
    
    # Tabs for different modes
    tab1, tab2, tab3 = st.tabs(["🔍 Single Match", "📋 Batch Analysis", "📈 Statistics"])
    
    # =============================
    # TAB 1: Single Match
    # =============================
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Select Patient")
            patient_id = st.selectbox(
                "Patient ID",
                sorted(patient_map.keys()),
                key="patient_select",
                label_visibility="collapsed"
            )
            patient = patient_map[patient_id]
            
            # Patient card
            st.markdown('<div class="patient-card">', unsafe_allow_html=True)
            summary = get_patient_summary(patient)
            st.markdown(f"**ID:** {summary['ID']}")
            st.markdown(f"**Age:** {summary['Age']} | **Gender:** {summary['Gender']}")
            st.markdown(f"**Conditions:** {summary['Conditions']}")
            if summary['Negated'] != "None":
                st.markdown(f"**No history of:** {summary['Negated']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if show_raw_text:
                with st.expander("📝 Raw Text"):
                    st.text(patient["raw_text"])
        
        with col2:
            st.markdown("### 🏥 Select Trial")
            trial_id = st.selectbox(
                "Trial ID",
                sorted(trial_map.keys()),
                key="trial_select",
                label_visibility="collapsed"
            )
            trial = trial_map[trial_id]
            
            # Trial card
            st.markdown('<div class="trial-card">', unsafe_allow_html=True)
            t_summary = get_trial_summary(trial)
            st.markdown(f"**ID:** {t_summary['ID']}")
            st.markdown(f"**Age Range:** {t_summary['Age Range']} years")
            st.markdown(f"**Required:** {t_summary['Required']}")
            st.markdown(f"**Excluded:** {t_summary['Excluded']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if show_raw_text:
                with st.expander("📝 Eligibility Text"):
                    st.text(trial["eligibility_text"])
        
        st.markdown("---")
        
        # Check Eligibility Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            check_button = st.button("🔬 Check Eligibility", use_container_width=True, type="primary")
        
        if check_button:
            # Rule-based analysis
            checks, rule_eligible = get_eligibility_details(patient, trial)
            
            # ML prediction
            if show_model_prediction and vectorizer is not None and classifier is not None:
                with st.spinner("Running ML prediction..."):
                    anon_text = anonymize(patient["raw_text"])
                    combined_text = anon_text + " " + trial["eligibility_text"]
                    processed_text = preprocess(combined_text)
                    X = vectorizer.transform([processed_text])
                    ml_prediction = classifier.predict(X)[0]
                    
                    if hasattr(classifier, "predict_proba"):
                        ml_score = classifier.predict_proba(X)[0][1]
                    else:
                        ml_score = float(ml_prediction)
            
            # Results
            st.markdown("---")
            st.markdown("## 🔍 Results")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.markdown("### Rule-Based")
                if rule_eligible:
                    st.success("✅ ELIGIBLE")
                else:
                    st.error("❌ NOT ELIGIBLE")
            
            if show_model_prediction:
                with res_col2:
                    st.markdown("### ML Prediction")
                    if ml_prediction == 1:
                        st.success("✅ ELIGIBLE")
                    else:
                        st.error("❌ NOT ELIGIBLE")
                
                with res_col3:
                    st.markdown("### Confidence")
                    st.metric("Score", f"{ml_score:.2%}")
                    st.progress(ml_score)
            
            # Detailed breakdown
            st.markdown("---")
            st.markdown("### 📋 Eligibility Criteria Breakdown")
            
            df = pd.DataFrame(checks)
            
            # Style the dataframe
            def highlight_status(val):
                if "✅" in str(val):
                    return "background-color: #C8E6C9"
                elif "❌" in str(val):
                    return "background-color: #FFCDD2"
                return ""
            
            styled_df = df.style.applymap(highlight_status, subset=["status"])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Agreement check
            if show_model_prediction:
                if (ml_prediction == 1) == rule_eligible:
                    st.info("✅ ML prediction agrees with rule-based analysis")
                else:
                    st.warning("⚠️ ML prediction differs from rule-based analysis")
    
    # =============================
    # TAB 2: Batch Analysis
    # =============================
    with tab2:
        st.markdown("### 📋 Find All Matching Trials for a Patient")
        
        batch_patient_id = st.selectbox(
            "Select Patient for Batch Analysis",
            sorted(patient_map.keys()),
            key="batch_patient"
        )
        
        if st.button("🔍 Find All Matches", type="primary"):
            batch_patient = patient_map[batch_patient_id]
            
            results = []
            for tid, trial in trial_map.items():
                checks, eligible = get_eligibility_details(batch_patient, trial)
                results.append({
                    "Trial ID": tid,
                    "Age Range": f"{trial['criteria']['min_age']}-{trial['criteria']['max_age']}",
                    "Required": ", ".join(trial['criteria']['required_conditions']),
                    "Eligible": "✅ Yes" if eligible else "❌ No"
                })
            
            results_df = pd.DataFrame(results)
            
            # Summary
            eligible_count = sum(1 for r in results if "✅" in r["Eligible"])
            col1, col2 = st.columns(2)
            col1.metric("Eligible Trials", eligible_count)
            col2.metric("Total Trials", len(results))
            
            st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # =============================
    # TAB 3: Statistics
    # =============================
    with tab3:
        st.markdown("### 📈 Dataset Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Age Distribution")
            ages = [p["metadata"]["age"] for p in patients]
            age_df = pd.DataFrame({"Age": ages})
            st.bar_chart(age_df["Age"].value_counts().sort_index())
        
        with col2:
            st.markdown("#### Gender Distribution")
            genders = [p["metadata"]["gender"] for p in patients]
            gender_counts = pd.Series(genders).value_counts()
            st.bar_chart(gender_counts)
        
        st.markdown("#### Top 10 Conditions")
        all_conditions = []
        for p in patients:
            all_conditions.extend(p["metadata"]["conditions"])
        cond_counts = pd.Series(all_conditions).value_counts().head(10)
        st.bar_chart(cond_counts)


if __name__ == "__main__":
    main()
