# app.py — Airbnb Price Prediction (Asheville-focused, detailed workflow)
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import plotly.express as px
import tensorflow as tf
from tensorflow.keras import layers, models

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Airbnb Price Prediction (Asheville, NC)", page_icon="🏠", layout="wide")
st.title("🏠 Airbnb Price Prediction — Asheville, NC (Detailed Workflow)")

# --- Step Header Styles (drop-in) ---
STEP_STYLES = """
<style>
.step-wrap { margin: 18px 0 8px 0; }
.step-chip {
  display: inline-block; padding: 4px 10px; border-radius: 999px;
  font-size: 12px; font-weight: 700; letter-spacing:.4px; text-transform: uppercase;
  color: white; background: var(--chip, #0ea5e9);
}
.step-title {
  margin: 6px 0 2px 0; font-size: 22px; font-weight: 800; color: #0b1220;
}
.step-sub {
  margin: 0; font-size: 14px; opacity: .8;
}
</style>
"""
st.markdown(STEP_STYLES, unsafe_allow_html=True)

def step_header(step_no: int, title: str, sub: str = "", color: str = "#0ea5e9"):
    """
    Renders a nice step heading with a colored pill.
    color: hex (e.g., "#0ea5e9" blue, "#10b981" green, "#f59e0b" amber, "#8b5cf6" violet)
    """
    html = f"""
    <div class="step-wrap" style="--chip:{color}">
      <div class="step-chip">Step {step_no}</div>
      <div class="step-title">{title}</div>
      {f'<div class="step-sub">{sub}</div>' if sub else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ----------------------------
# Helpers
# ----------------------------
DATA_PATH = Path("listings.csv")  # Hardcoded
TARGET = "price"

NUMERIC_KEEP = ["price", "bathrooms", "bedrooms", "number_of_reviews", "latitude", "longitude"]
CATEGORICAL_KEEP = ["room_type", "host_identity_verified", "host_is_superhost"]

def clean_price_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def to_bool_t_f(s: pd.Series) -> pd.Series:
    # Convert 't'/'f' strings to boolean where present
    if s.dropna().astype(str).str.lower().isin(["t", "f"]).all():
        return s.astype(str).str.lower().map({"t": True, "f": False})
    return s

def percentile_cap(s: pd.Series, q=0.99) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(np.nanpercentile(s.dropna(), q * 100)) if s.notna().any() else np.nan

# ====================================================
# Section 1 — Data Loading
# ====================================================
step_header(1, "Data Loading", "Import Inside Airbnb listings & sanity-check schema", color="#0ea5e9")

if not DATA_PATH.exists():
    st.error("❌ File not found: `listings.csv`. Please add the dataset and rerun.")
    st.stop()

df_raw = pd.read_csv(DATA_PATH, low_memory=False)
st.success(f"✅ Loaded {df_raw.shape[0]:,} rows × {df_raw.shape[1]:,} columns")
st.caption("Preview (first 5 rows)")
st.dataframe(df_raw.head(), use_container_width=True)

# ====================================================
# Section 2 — Data Selection (Asheville, NC only)
# ====================================================
st.header("Section 2 — Data Selection (Asheville, NC)")

df = df_raw.copy()

# --- 2.1 Robust Asheville, NC filter ---
def asheville_mask(frame: pd.DataFrame) -> pd.Series:
    n = len(frame)
    mask_any = pd.Series(False, index=frame.index)

    # Text-based city/neighbourhood matches
    def has(col): return col in frame.columns
    def contains(col, pat): 
        return frame[col].astype(str).str.contains(pat, case=False, na=False) if has(col) else pd.Series(False, index=frame.index)

    city_ashe = contains("city", r"\bAsheville\b")
    market_ashe = contains("market", r"\bAsheville\b")
    neigh_ashe = contains("neighbourhood_cleansed", r"\bAsheville\b") | contains("neighbourhood", r"\bAsheville\b")
    text_ashe = city_ashe | market_ashe | neigh_ashe

    # NC constraint if state exists; otherwise allow
    if has("state"):
        state_nc = frame["state"].astype(str).str.upper().str.contains(r"\bNC\b|NORTH CAROLINA", na=False)
    else:
        state_nc = pd.Series(True, index=frame.index)  # no state column: don't block

    mask_text = text_ashe & state_nc

    # Geo fallback (if no rows from text filter and lat/lon available)
    if (not mask_text.any()) and has("latitude") and has("longitude"):
        # Loose bounding box around Asheville
        lat = pd.to_numeric(frame["latitude"], errors="coerce")
        lon = pd.to_numeric(frame["longitude"], errors="coerce")
        geo_box = (lat.between(35.40, 35.82)) & (lon.between(-82.75, -82.25))
        mask_any = geo_box
    else:
        mask_any = mask_text

    return mask_any

mask_ashe = asheville_mask(df)

if mask_ashe.any():
    kept = int(mask_ashe.sum())
    st.info(f"Filtered to **Asheville, NC**: kept **{kept:,}** listings.")
    df = df[mask_ashe].copy()
else:
    st.warning("Could not confidently isolate Asheville via available fields; proceeding with the entire dataset (no filter applied).")

# --- 2.2 Keep compact, useful subset (numeric + categorical) ---
present_numeric = [c for c in NUMERIC_KEEP if c in df.columns]
present_cats    = [c for c in CATEGORICAL_KEEP if c in df.columns]

# Ensure target present first, then selected cols
keep_cols = list(dict.fromkeys([TARGET] + present_numeric + present_cats))
df = df[keep_cols].copy()

st.subheader("Columns kept for analysis")

# Two-column table — Numeric | Categorical (styled)
def two_col_table(n_list, c_list):
    m = max(len(n_list), len(c_list))
    n_list = n_list + [""] * (m - len(n_list))
    c_list = c_list + [""] * (m - len(c_list))
    return pd.DataFrame({"Numeric": n_list, "Categorical": c_list})

cols_tbl = two_col_table(present_numeric, present_cats)
st.dataframe(
    cols_tbl.style.set_properties(**{"text-align": "left"}),
    use_container_width=True
)

st.caption("Preview of the working dataframe")
st.dataframe(df.head(), use_container_width=True)

st.caption("We keep a compact set of numeric & categorical fields to balance model signal and generality across city exports.")

# ====================================================
# Section 3 — Data Preprocessing
# ====================================================
step_header(3, "Data Preprocessing", "Drop missing price → fix types → ensure only superhost is missing → mode-impute", color="#f59e0b")

st.markdown("""
We enforce the following sequence (to mirror the Colab):
1) **Drop** rows with missing **price** (we never impute the target).
2) **Coerce** key numerics to numeric and **drop** any rows where those are still missing — so that **only** `host_is_superhost` remains missing.
3) **Mode-impute** `host_is_superhost`.
4) Fix types (convert `'t'/'f'` → `True/False`; keep `room_type` as categorical).
""")

# 3.1 — Drop rows with missing price (no target imputation)
df[TARGET] = clean_price_series(df[TARGET])
before_drop_price = len(df)
df = df.dropna(subset=[TARGET]).copy()
after_drop_price = len(df)
st.info(f"🧹 Dropped **{before_drop_price - after_drop_price}** rows with missing `price`; kept **{after_drop_price}** rows.")

# 3.2 — Coerce numerics & drop rows with missing numerics so only 'host_is_superhost' remains missing
core_numeric = [c for c in ["bathrooms", "bedrooms", "number_of_reviews", "latitude", "longitude"] if c in df.columns]

# coerce to numeric (this can create NaN when cells have non-numeric strings)
for c in core_numeric:
    df[c] = pd.to_numeric(df[c], errors="coerce")

before_drop_num = len(df)
df = df.dropna(subset=core_numeric).copy()
after_drop_num = len(df)
if before_drop_num != after_drop_num:
    st.caption(f"Ensured clean numerics: dropped **{before_drop_num - after_drop_num}** rows with missing {core_numeric}")

# 3.3 — Show missingness *now* (after dropping price + numeric cleanup)
st.subheader("Missingness after dropping rows with missing price & cleaning numerics")
miss_now = (df.isna().mean() * 100).round(2).to_frame("missing_%")
st.dataframe(miss_now.style.format({"missing_%": "{:.2f}"}), use_container_width=True)

st.markdown("""
At this point, the only expected missingness should be in **`host_is_superhost`** (per Colab).
We now fill it using **mode imputation** (most frequent value).
""")

# 3.4 — Mode imputation for host_is_superhost
if "host_is_superhost" in df.columns:
    mode_val = df["host_is_superhost"].mode(dropna=True)
    if not mode_val.empty:
        fill_value = mode_val.iloc[0]
        df["host_is_superhost"] = df["host_is_superhost"].fillna(fill_value)
        st.success(f"Filled missing `host_is_superhost` with mode: **{fill_value!r}**")
    else:
        st.warning("Could not compute a mode for `host_is_superhost` (no non-null values).")
else:
    st.info("`host_is_superhost` not present in the dataset.")

# 3.5 — Type fixes
for c in ["host_identity_verified", "host_is_superhost"]:
    if c in df.columns:
        df[c] = to_bool_t_f(df[c])  # 't'/'f' → True/False

if "room_type" in df.columns:
    df["room_type"] = df["room_type"].astype("category")

# 3.6 — Missingness after imputation (should be all ~0%)
st.subheader("Missingness after mode-imputation & type fixes")
miss_final = (df.isna().mean() * 100).round(2).to_frame("missing_%")
st.dataframe(miss_final.style.format({"missing_%": "{:.2f}"}), use_container_width=True)

# ====================================================
# Section 4 — Exploratory Data Analysis (EDA)
# ====================================================
st.header("Section 4 — Exploratory Data Analysis (EDA)")

# 4.1: Descriptive statistics
st.subheader("4.1 Descriptive Statistics (Numeric)")
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if numeric_cols:
    st.dataframe(df[numeric_cols].describe().T.round(3), use_container_width=True)
else:
    st.info("No numeric columns available for describe().")

# 4.2: Histograms
st.subheader("4.2 Histograms of Distributions")
h_cols = [c for c in ["price", "bedrooms", "bathrooms", "number_of_reviews"] if c in df.columns]
if h_cols:
    for c in h_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[c].dropna(), bins=40, ax=ax, kde=False)
        ax.set_xlabel(c)
        ax.set_ylabel("Count")
        st.pyplot(fig)
else:
    st.info("No standard numeric columns found for histograms.")

# 4.3: Pair Plot (sampled for performance)
st.subheader("4.3 Pair Plot")
pp_cols = [c for c in ["price", "bedrooms", "bathrooms", "number_of_reviews"] if c in df.columns]
if len(pp_cols) >= 2:
    sample_df = df[pp_cols].dropna().sample(min(1000, df.shape[0]), random_state=42)
    fig = sns.pairplot(sample_df, diag_kind="hist")
    st.pyplot(fig)
else:
    st.info("Not enough numeric columns for a pair plot.")

# 4.4: Correlation Matrix
st.subheader("4.4 Correlation Matrix (Numeric)")
if len(numeric_cols) >= 2:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(numeric_only=True), cmap="YlGnBu", ax=ax)
    st.pyplot(fig)
else:
    st.info("Not enough numeric features for correlation matrix.")

# 4.5: Handling Outliers (99th percentile cap for price)
st.subheader("4.5 Handling Outliers — 99th Percentile Rule (price)")
if "price" in df.columns and df["price"].notna().any():
    cap = percentile_cap(df["price"], q=0.99)
    st.caption(f"Applying cap at 99th percentile: price <= {cap:,.2f}")
    df_wo = df[df["price"] <= cap].copy()

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.histplot(df["price"], bins=40, ax=ax, color="#ef4444")
        ax.set_title("Before outlier handling")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        sns.histplot(df_wo["price"], bins=40, ax=ax, color="#10b981")
        ax.set_title("After outlier handling (<= 99th pct)")
        st.pyplot(fig)
else:
    st.info("No price column for outlier handling.")
    df_wo = df.copy()

# 4.6: Geospatial Plot (Map with price buckets)
st.subheader("4.6 Geospatial Distribution (price buckets)")
if set(["latitude", "longitude", "price"]).issubset(df_wo.columns):
    buckets = pd.cut(
        df_wo["price"],
        bins=[-np.inf, 100, 200, 300, np.inf],
        labels=["< 100", "100–200", "200–300", "> 300"]
    )
    df_map = df_wo.assign(price_bucket=buckets)

    fig = px.scatter_mapbox(
        df_map,
        lat="latitude",
        lon="longitude",
        color="price_bucket",
        hover_data=["price"] + [c for c in ["bedrooms", "bathrooms"] if c in df_map.columns],
        zoom=10,
        height=500,
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Latitude/Longitude/Price not all available for a geospatial plot.")

# ====================================================
# Section 5 — Prepare for Modeling (Normalize & Split) + Keras Model
# ====================================================
st.header("Section 5 — Prepare for Modeling & Keras Training")

# 5.1 Feature selection for modeling
model_num = [c for c in ["bedrooms", "bathrooms", "number_of_reviews", "latitude", "longitude"] if c in df_wo.columns]
model_cat = [c for c in ["room_type", "host_identity_verified", "host_is_superhost"] if c in df_wo.columns]

work = df_wo[[TARGET] + model_num + model_cat].dropna().copy()
y = work[TARGET].astype(float)
X_num = work[model_num].copy()

# Encode categoricals with pandas.get_dummies (simple, stable)
if model_cat:
    X_cat = pd.get_dummies(work[model_cat].astype("category"), drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)
else:
    X = X_num

# Train-test split (80/20, seed=42)
X_train, X_test, y_train, y_test = train_test_split(
    X.values.astype("float32"),
    y.values.astype("float32"),
    test_size=0.2,
    random_state=42
)

# 5.2 Build Keras model with Normalization
norm = layers.Normalization()
norm.adapt(X_train)  # learns mean/variance from training data

model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    norm,
    layers.Dense(64, activation="relu"),
    layers.Dense(1)  # regression output
])

model.compile(
    loss="mse",
    optimizer="adam",
    metrics=["mae"]
)

# 5.3 Train the model (20% of training as validation, 50 epochs)
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=0
)

# ---- Modeling Graphs (before results) ----
st.subheader("Training Curves (Loss & MAE)")

hist = pd.DataFrame(history.history)
c1, c2 = st.columns(2)

with c1:
    fig, ax = plt.subplots()
    ax.plot(hist.index, hist["loss"], label="Train Loss (MSE)")
    ax.plot(hist.index, hist["val_loss"], label="Val Loss (MSE)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE)"); ax.legend()
    st.pyplot(fig)

with c2:
    fig, ax = plt.subplots()
    ax.plot(hist.index, hist["mae"], label="Train MAE")
    ax.plot(hist.index, hist["val_mae"], label="Val MAE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE"); ax.legend()
    st.pyplot(fig)

# Predictions for test set
y_pred = model.predict(X_test, verbose=0).ravel()

st.subheader("Predicted vs Actual (Test)")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.4, color="#0ea5e9")
lims = [float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual Price"); ax.set_ylabel("Predicted Price")
st.pyplot(fig)

# ====================================================
# FINAL RESULTS (after all graphs)
# ====================================================
st.header("Final Results")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))
r2 = r2_score(y_test, y_pred)

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:,.2f}")
m2.metric("RMSE", f"{rmse:,.2f}")
m3.metric("R²", f"{r2:.3f}")

st.caption("Notes: MAE/RMSE reflect average error; R² measures explained variance. Training curves show convergence; map highlights geographic pricing patterns.")
