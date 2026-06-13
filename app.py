"""
Crop Recommendation System — Streamlit App
==========================================
Run:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size: 2.2rem; font-weight: 700; color: #0F6E56; margin-bottom: 0; }
    .subtitle    { font-size: 1rem;   color: #6b7280; margin-top: 0.2rem; margin-bottom: 1.5rem; }
    .section-hdr { font-size: 1rem;   font-weight: 600; color: #374151; margin-bottom: 0.5rem; }
    .metric-card {
        background: #f0faf5; border-left: 4px solid #1D9E75;
        border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
    }
    .crop-result {
        background: linear-gradient(135deg, #e8f5e9, #f0fdf4);
        border: 2px solid #1D9E75; border-radius: 12px;
        padding: 1.5rem; text-align: center; margin-bottom: 1rem;
    }
    .crop-name   { font-size: 2.4rem; font-weight: 800; color: #0F6E56; }
    .crop-conf   { font-size: 1rem;   color: #6b7280; }
    .alt-chip    {
        display: inline-block; background: #f0faf5; border: 1px solid #1D9E75;
        border-radius: 20px; padding: 4px 14px; margin: 4px;
        font-size: 0.88rem; color: #0F6E56;
    }
</style>
""", unsafe_allow_html=True)

# ── Load or Train models ───────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    model_dir = os.path.join(BASE, 'models')
    os.makedirs(model_dir, exist_ok=True)

    rf_path     = os.path.join(model_dir, 'rf_model.pkl')
    knn_path    = os.path.join(model_dir, 'knn_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    le_path     = os.path.join(model_dir, 'label_encoder.pkl')

    # If models already saved, load them
    if all(os.path.exists(p) for p in [rf_path, knn_path, scaler_path, le_path]):
        rf     = joblib.load(rf_path)
        knn    = joblib.load(knn_path)
        scaler = joblib.load(scaler_path)
        le     = joblib.load(le_path)
        return rf, knn, scaler, le

    # Otherwise train from scratch using the CSV
    data_path = os.path.join(BASE, 'Crop_recommendation.csv')
    df = pd.read_csv(data_path)

    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    le = LabelEncoder()
    X  = df[features].values
    y  = le.fit_transform(df['label'].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_sc, y_train)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_sc, y_train)

    # Save for next time
    joblib.dump(rf,     rf_path)
    joblib.dump(knn,    knn_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le,     le_path)

    return rf, knn, scaler, le

rf, knn, scaler, le = load_models()

FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# ── Crop info dict ────────────────────────────────────────────────────────────
CROP_INFO = {
    "rice":        {"emoji": "🌾", "season": "Kharif",  "water": "High",   "tip": "Grows best in flooded paddies with standing water."},
    "wheat":       {"emoji": "🌾", "season": "Rabi",    "water": "Medium", "tip": "Cool weather crop; sow after monsoon withdraws."},
    "maize":       {"emoji": "🌽", "season": "Kharif",  "water": "Medium", "tip": "Well-drained soil is essential; avoid waterlogging."},
    "chickpea":    {"emoji": "🫘", "season": "Rabi",    "water": "Low",    "tip": "Drought-tolerant legume; fixes atmospheric nitrogen."},
    "kidneybeans": {"emoji": "🫘", "season": "Kharif",  "water": "Medium", "tip": "Requires well-drained, fertile loamy soil."},
    "pigeonpeas":  {"emoji": "🫘", "season": "Kharif",  "water": "Low",    "tip": "Drought-tolerant; improves soil with nitrogen fixation."},
    "mothbeans":   {"emoji": "🫘", "season": "Kharif",  "water": "Low",    "tip": "Extremely drought-resistant; ideal for arid zones."},
    "mungbean":    {"emoji": "🫘", "season": "Kharif",  "water": "Low",    "tip": "Short-duration crop; enriches soil naturally."},
    "blackgram":   {"emoji": "🫘", "season": "Kharif",  "water": "Low",    "tip": "Tolerates light drought; needs well-drained soil."},
    "lentil":      {"emoji": "🫘", "season": "Rabi",    "water": "Low",    "tip": "Cool-season legume; excellent soil builder."},
    "pomegranate": {"emoji": "🍎", "season": "Year-round","water": "Low",  "tip": "Very drought-tolerant fruit; prefers dry climates."},
    "banana":      {"emoji": "🍌", "season": "Year-round","water": "High", "tip": "Needs consistent moisture; wind protection advised."},
    "mango":       {"emoji": "🥭", "season": "Summer",  "water": "Low",    "tip": "Dry spell before flowering boosts fruit yield."},
    "grapes":      {"emoji": "🍇", "season": "Rabi",    "water": "Medium", "tip": "Well-drained sandy loam; trellising required."},
    "watermelon":  {"emoji": "🍉", "season": "Zaid",    "water": "Medium", "tip": "Hot-weather crop; large space between plants needed."},
    "muskmelon":   {"emoji": "🍈", "season": "Zaid",    "water": "Medium", "tip": "Sandy loam soil; avoid over-watering to prevent rot."},
    "apple":       {"emoji": "🍎", "season": "Rabi",    "water": "Medium", "tip": "Requires chilling hours; hilly areas are best."},
    "orange":      {"emoji": "🍊", "season": "Winter",  "water": "Medium", "tip": "Subtropical climate; avoid frost zones."},
    "papaya":      {"emoji": "🥭", "season": "Year-round","water": "Medium","tip": "Fast-growing; cannot withstand frost or waterlogging."},
    "coconut":     {"emoji": "🥥", "season": "Year-round","water": "High", "tip": "Coastal sandy soils; high humidity is beneficial."},
    "cotton":      {"emoji": "🌿", "season": "Kharif",  "water": "Medium", "tip": "Black cotton soil is ideal; needs 180–200 frost-free days."},
    "jute":        {"emoji": "🌿", "season": "Kharif",  "water": "High",   "tip": "Alluvial soil near rivers; high humidity required."},
    "coffee":      {"emoji": "☕", "season": "Year-round","water": "Medium","tip": "Shade-grown on hillsides; acidic soil preferred."},
}

def get_info(crop):
    return CROP_INFO.get(crop.lower(), {"emoji":"🌱","season":"Varies","water":"Medium","tip":""})

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🌾 Crop Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter your soil and climate data to get AI-powered crop recommendations</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🌱 Recommend a Crop", "📊 Data Insights", "📋 Model Info"])

# ═══════════════════════════════════════════════════════
# TAB 1 — RECOMMENDATION
# ═══════════════════════════════════════════════════════
with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### 🧪 Soil Nutrients")
        c1, c2, c3 = st.columns(3)
        N = c1.number_input("Nitrogen (N) mg/kg", 0, 200, 90, help="Soil nitrogen content")
        P = c2.number_input("Phosphorus (P) mg/kg", 0, 200, 42, help="Soil phosphorus content")
        K = c3.number_input("Potassium (K) mg/kg", 0, 200, 43, help="Soil potassium content")
        ph = st.slider("Soil pH", 0.0, 14.0, 6.5, 0.1)
        st.markdown(f"**pH level:** {'Acidic' if ph < 6 else 'Neutral' if ph <= 7.5 else 'Alkaline'} ({ph})")

        st.markdown("### 🌤️ Climate Conditions")
        c4, c5 = st.columns(2)
        temperature = c4.number_input("Temperature (°C)", -10, 55, 25)
        humidity    = c5.number_input("Humidity (%)", 0, 100, 80)
        rainfall    = st.number_input("Annual Rainfall (mm)", 0, 5000, 200)

        st.markdown("### 🤖 Model")
        model_choice = st.radio("Choose ML model", ["Random Forest (recommended)", "KNN"], horizontal=True)

        predict_btn = st.button("🌱 Get Crop Recommendation", use_container_width=True, type="primary")

    with col_result:
        if predict_btn:
            sample    = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            sample_sc = scaler.transform(sample)

            model    = rf if "Random" in model_choice else knn
            pred_enc = model.predict(sample_sc)[0]
            crop     = le.inverse_transform([pred_enc])[0]
            proba    = model.predict_proba(sample_sc)[0]
            conf     = proba[pred_enc] * 100
            info     = get_info(crop)

            top3_idx = np.argsort(proba)[::-1][:4]
            alts     = [le.classes_[i] for i in top3_idx if le.classes_[i] != crop][:3]

            st.markdown(f"""
            <div class="crop-result">
                <div style="font-size:3rem">{info['emoji']}</div>
                <div class="crop-name">{crop.title()}</div>
                <div class="crop-conf">Confidence: {conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(conf))

            c_a, c_b = st.columns(2)
            c_a.metric("💧 Water Need",  info["water"])
            c_b.metric("📅 Best Season", info["season"])

            if info["tip"]:
                st.info(f"**Tip:** {info['tip']}")

            st.markdown("**Alternative crops:**")
            chips = " ".join([f'<span class="alt-chip">{a.title()}</span>' for a in alts])
            st.markdown(chips, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Input Summary**")
            summary = pd.DataFrame({
                "Feature": ["N (mg/kg)", "P (mg/kg)", "K (mg/kg)", "Temperature (°C)",
                             "Humidity (%)", "pH", "Rainfall (mm)"],
                "Value":   [N, P, K, temperature, humidity, ph, rainfall]
            })
            st.dataframe(summary.set_index("Feature"), use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem 1rem; color:#9ca3af;">
                <div style="font-size:3rem">🌱</div>
                <p>Fill in your soil & climate data<br>then click <strong>Get Crop Recommendation</strong></p>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# TAB 2 — DATA INSIGHTS
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Dataset Overview")

    data_path = os.path.join(BASE, 'Crop_recommendation.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Samples",  len(df))
        m2.metric("Unique Crops",   df['label'].nunique())
        m3.metric("Features",       len(FEATURES))
        m4.metric("Missing Values", int(df.isnull().sum().sum()))

        st.dataframe(df.describe().round(2), use_container_width=True)

        st.markdown("### Feature Distributions")
        plot_dir = os.path.join(BASE, 'plots')
        plots = {
            "Feature Distributions": "feature_dist.png",
            "Correlation Heatmap":   "correlation.png",
            "Samples per Crop":      "crop_counts.png",
            "Feature Importance":    "feature_importance.png",
            "Model Comparison":      "model_comparison.png",
        }
        for title, fname in plots.items():
            fpath = os.path.join(plot_dir, fname)
            if os.path.exists(fpath):
                st.markdown(f"**{title}**")
                st.image(fpath, use_container_width=True)
                st.markdown("---")

        # Confusion matrices
        for label, fname in [("Random Forest", "confusion_matrix_rf.png"),
                              ("KNN", "confusion_matrix_knn.png")]:
            fpath = os.path.join(plot_dir, fname)
            if os.path.exists(fpath):
                st.markdown(f"**{label} — Confusion Matrix**")
                st.image(fpath, use_container_width=True)
                st.markdown("---")
    else:
        st.warning("Dataset not found. Run `train_model.py` first.")


# ═══════════════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🤖 Model Information")

    col_rf, col_knn = st.columns(2)

    with col_rf:
        st.markdown("#### Random Forest")
        st.markdown("""
| Parameter | Value |
|-----------|-------|
| n_estimators | 100 |
| max_features | sqrt |
| criterion | gini |
| random_state | 42 |
| Test Accuracy | **~90.65%** |
""")
        st.success("✅ Recommended model")

    with col_knn:
        st.markdown("#### K-Nearest Neighbors")
        st.markdown("""
| Parameter | Value |
|-----------|-------|
| n_neighbors | 5 |
| weights | uniform |
| algorithm | auto |
| metric | minkowski |
| Test Accuracy | **~84.57%** |
""")

    st.markdown("---")
    st.markdown("### 🔄 Pipeline")
    st.markdown("""
```
Raw Input (7 features)
    ↓
StandardScaler (normalize to mean=0, std=1)
    ↓
Random Forest Classifier (100 trees)
    ↓
Predicted Crop + Probability Scores
```
    """)

    st.markdown("### 📦 Features Used")
    feat_df = pd.DataFrame({
        "Feature": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        "Description": [
            "Nitrogen content in soil (mg/kg)",
            "Phosphorus content in soil (mg/kg)",
            "Potassium content in soil (mg/kg)",
            "Average temperature (°C)",
            "Relative humidity (%)",
            "Soil pH value",
            "Annual rainfall (mm)",
        ]
    })
    st.dataframe(feat_df.set_index("Feature"), use_container_width=True)

    st.markdown("### 🌱 Supported Crops (23)")
    crops = [c.title() for c in sorted(le.classes_)]
    cols  = st.columns(4)
    for i, crop in enumerate(crops):
        info = get_info(crop.lower())
        cols[i % 4].markdown(f"{info['emoji']} {crop}")
