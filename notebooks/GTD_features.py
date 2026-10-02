import numpy as np
import pandas as pd

NUM_INPUTS = ["iyear", "imonth", "suicide", "multiple", "extended", "individual"]
FREQ_INPUTS = {"country_txt": "country_txt_freq", "gname": "gname_freq"}
DUMMY_SOURCES = ["region_txt", "attacktype1_txt", "targtype1_txt", "weaptype1_txt"]


def _scale(scaler, col, values):
    i = list(scaler.feature_names_in_).index(col)
    return (values - scaler.mean_[i]) / scaler.scale_[i]


def unscale(scaler, col, values):
    """Inverse of Step 4's scaling — used to recover raw values such as nkill."""
    i = list(scaler.feature_names_in_).index(col)
    return values * scaler.scale_[i] + scaler.mean_[i]


def build_features(rows: pd.DataFrame, art: dict) -> pd.DataFrame:
    """rows: raw values with columns NUM_INPUTS + FREQ_INPUTS keys + DUMMY_SOURCES.
    art: the dict saved as preprocessor.pkl. Returns the model's feature matrix."""
    scaler = art["scaler"]
    out = pd.DataFrame(0.0, index=rows.index, columns=art["feature_cols"])

    for col in NUM_INPUTS:
        out[col] = _scale(scaler, col, rows[col].astype(float))

    for raw_col, enc_col in FREQ_INPUTS.items():
        freq = rows[raw_col].map(art["freq_maps"][raw_col])
        freq = freq.fillna(art["freq_floor"][raw_col])   # unseen category -> rarest
        out[enc_col] = _scale(scaler, enc_col, freq)

    for src in DUMMY_SOURCES:
        for idx, val in rows[src].items():
            name = f"{src}_{val}"
            if name in out.columns:
                out.loc[idx, name] = 1.0
    return out