import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "notebooks"))  
from notebooks.GTD_features import build_features  

st.set_page_config(page_title="GTD Fatality Predictor", page_icon="📊", layout="centered")

MONTHS = ["Unknown", "January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "models" / "best_model.pkl")
    prep = joblib.load(BASE_DIR / "models" / "preprocessor.pkl")
    with open(BASE_DIR / "app_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    return model, prep, cfg


model, prep, cfg = load_artifacts()
opts = cfg["options"]

st.title("Will this incident be fatal?")
st.write(
    "Describe an incident using the fields below. The model — trained on the "
    "Global Terrorism Database (1970–2017) — estimates the probability that "
    "at least one person was killed."
)

col1, col2 = st.columns(2)
with col1:
    year = st.slider("Year", cfg["year_min"], cfg["year_max"], 2010)
    month_name = st.selectbox("Month", MONTHS, index=0)
    region = st.selectbox("Region", opts["region_txt"])
    country = st.selectbox("Country", cfg["region_to_countries"][region])
with col2:
    attack = st.selectbox("Attack type", opts["attacktype1_txt"])
    target = st.selectbox("Target type", opts["targtype1_txt"])
    weapon = st.selectbox("Weapon type", opts["weaptype1_txt"])
    group = st.selectbox("Perpetrator group", opts["gname"] + ["Other group"])

st.markdown("**Incident characteristics**")
c1, c2, c3, c4 = st.columns(4)
suicide = c1.checkbox("Suicide attack")
multiple = c2.checkbox("Part of multiple attacks")
extended = c3.checkbox("Lasted > 24 hours")
individual = c4.checkbox("Lone individual")

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{
        "region_txt": region,
        "country_txt": country,
        "attacktype1_txt": attack,
        "targtype1_txt": target,
        "weaptype1_txt": weapon,
        "gname": group,
        "iyear": year,
        "imonth": MONTHS.index(month_name),
        "suicide": int(suicide),
        "multiple": int(multiple),
        "extended": int(extended),
        "individual": int(individual),
    }])

    features = build_features(row, prep).astype("float32")
    proba = float(model.predict_proba(features)[0, 1])
    fatal = proba >= 0.5

    st.divider()
    if fatal:
        st.error("Prediction: **FATAL** incident likely")
    else:
        st.success("Prediction: **NON-FATAL** incident likely")
    st.metric("Probability of at least one death", f"{proba * 100:.1f}%")
    st.progress(proba)

st.caption(
    f"Model: {cfg['best_model']} · test accuracy "
    f"{cfg['test_metrics']['Accuracy'] * 100:.1f}% · F1 {cfg['test_metrics']['F1']:.3f}. "
    "This reflects statistical patterns in historical records, not a real-world threat assessment."
)