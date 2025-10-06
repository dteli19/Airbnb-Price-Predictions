# app.py — Airbnb Price Prediction (Asheville-focused, detailed workflow, beautified)
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
.tbl-note { font-size: 13px; opacity: .9; margin: 6px 0 2px; }
hr.soft { border: none; border-top: 1px solid rgba(0,0,0,.05); margin: 6px 0 12px; }
.metric-card {
  border-radius:16px; padding:14px 16px; color:white;
  box-shadow:0 6px 20px rgba(0,0,0,.08);
}
</style>
"""
st.markdown(STEP_STYLES, unsafe_allow_html=True)

def step_header(step_no: int, title: str, sub: str = "", color: str = "#0ea5e9"):
    html = f"""
    <div class="step-wrap" style="--chip:{color}">
      <div class="step-chip">Step {step_no}</div>
      <div class="step-title">{title}</div>
      {f'<div class="step-sub">{sub}</div>' if sub else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def metric_card(title, value, subtitle=None, bg="#0ea5e9"):
    st.markdown(
        f"""
        <div class="metric-card" style="background:{bg}">
          <div style="font-size:12px;letter-spacing:.4px;text-transform:uppercase;opacity:.95">{title}</div>
          <div style="font-size:28px;font-weight:800;margin-top:4px">{value}</div>
          <div style="font-size:12px;opacity:.9;margin-top:2px">{subtitle or ""}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# Helpers
# ----------------------------
DATA_PATH = Path("listings.csv")  # Hardcoded (adjust to "data/listings.csv" if you prefer)
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

# ----------------------------------------------------
Overview, Problem, About the Data, Actions
# ----------------------------------------------------
st.markdown("""
### **Overview**
This project develops a complete machine learning pipeline to predict Airbnb listing prices in Asheville, NC, using real-world data from the Inside Airbnb dataset. The workflow covers all major stages of data science — from data cleaning and exploration to feature preparation and neural network modeling.
By analyzing host, property, and location attributes, the project uncovers the key factors that influence pricing and builds a baseline predictive model to estimate nightly rates accurately.

### **Problem Statement**
Estimate nightly **price** using listing attributes (capacity, room/property type, reviews, location) and surface patterns that explain price variation.

### **About the Data**
Source: *Inside Airbnb `listings.csv`* (city export). Typical fields include:
- **Numerical**: `price`, `bedrooms`, `bathrooms`, `number_of_reviews`, `latitude`, `longitude`
- **Categorical**: `room_type`, `host_identity_verified`, `host_is_superhost`
- (Columns vary by city; the pipeline adapts to what’s available.)

### **Actions & 5-Step Workflow**
1. **Data Loading** — Read `listings.csv`, preview rows, verify schema  
2. **Data Selection (Asheville, NC)** — Filter to city, keep a compact set of num/cat features  
3. **Data Preprocessing** — Drop missing `price`, clean numerics, **mode-impute** `host_is_superhost`, fix types  
4. **EDA** — Describe stats, distributions, pair plot, correlation, **outlier handling (99th pct)**, **geo map**  
5. **Modeling** — Train/test split (80/20), Keras model with **Normalization**, training curves, predictions → **Final Results**
""")


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
st.markdown('<hr class="soft"/>', unsafe_allow_html=True)

# ====================================================
# Section 2 — Data Selection (Asheville, NC only)
# ====================================================
step_header(2, "Data Selection (Asheville, NC)", "Filter to Asheville, then keep compact numeric/categorical sets", color="#10b981")

df = df_raw.copy()

# --- 2.1 Robust Asheville, NC filter ---
def asheville_mask(frame: pd.DataFrame) -> pd.Series:
    mask_any = pd.Series(False, index=frame.index)
    def has(col): return col in frame.columns
    def contains(col, pat): 
        return frame[col].astype(str).str.contains(pat, case=False, na=False) if has(col) else pd.Series(False, index=frame.index)

    city_ashe  = contains("city", r"\bAsheville\b")
    market_ashe = contains("market", r"\bAsheville\b")
    neigh_ashe = contains("neighbourhood_cleansed", r"\bAsheville\b") | contains("neighbourhood", r"\bAsheville\b")
    text_ashe = city_ashe | market_ashe | neigh_ashe

    # NC constraint if state exists; otherwise allow
    if has("state"):
        state_nc = frame["state"].astype(str).str.upper().str.contains(r"\bNC\b|NORTH CAROLINA", na=False)
    else:
        state_nc = pd.Series(True, index=frame.index)

    mask_text = text_ashe & state_nc

    # Geo fallback (if text filter yields none)
    if (not mask_text.any()) and has("latitude") and has("longitude"):
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
    st.warning("Could not confidently isolate Asheville via available fields; proceeding with the **entire dataset** (no filter applied).")

# --- 2.2 Keep compact, useful subset (numeric + categorical) ---
present_numeric = [c for c in NUMERIC_KEEP if c in df.columns]
present_cats    = [c for c in CATEGORICAL_KEEP if c in df.columns]

keep_cols = list(dict.fromkeys([TARGET] + present_numeric + present_cats))
df = df[keep_cols].copy()

# Small metrics row
mc1, mc2, mc3 = st.columns(3)
with mc1:
    metric_card("Listings", f"{len(df):,}", "After Asheville filter", bg="#0ea5e9")
with mc2:
    metric_card("Numeric cols", f"{len(present_numeric)}", ", ".join(present_numeric) if present_numeric else "—", bg="#10b981")
with mc3:
    metric_card("Categorical cols", f"{len(present_cats)}", ", ".join(present_cats) if present_cats else "—", bg="#8b5cf6")

st.subheader("Columns kept for analysis")

def two_col_table(n_list, c_list):
    m = max(len(n_list), len(c_list))
    n_list = n_list + [""] * (m - len(n_list))
    c_list = c_list + [""] * (m - len(c_list))
    return pd.DataFrame({"Numeric": n_list, "Categorical": c_list})

cols_tbl = two_col_table(present_numeric, present_cats)

# Render a styled HTML table (st.dataframe ignores Pandas Styler CSS)
def render_table_html(df, header_bg="#0b1620", header_fg="#ffffff",
                      even="#0f172a", odd="#111827", font="#e5e7eb"):
    # Build zebra rows
    rows_html = []
    for i, (_, row) in enumerate(df.iterrows()):
        bg = even if i % 2 == 0 else odd
        row_html = "".join([f'<td style="padding:8px 10px;color:{font}">{str(val) if val!="" else "&nbsp;"}</td>'
                            for val in row.values])
        rows_html.append(f'<tr style="background:{bg}">{row_html}</tr>')
    rows_html = "\n".join(rows_html)

    html = f"""
    <div style="border-radius:12px;overflow:hidden;border:1px solid rgba(0,0,0,.08);box-shadow:0 6px 20px rgba(0,0,0,.06)">
      <table style="border-collapse:collapse;width:100%">
        <thead>
          <tr style="background:{header_bg};color:{header_fg}">
            <th style="text-align:left;padding:10px 12px">Numeric</th>
            <th style="text-align:left;padding:10px 12px">Categorical</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    """
    return html

st.markdown(render_table_html(cols_tbl), unsafe_allow_html=True)

st.caption("A compact, portable feature set — enough signal for modeling while staying robust across different city exports.")

# ====================================================
# Section 3 — Data Preprocessing
# ====================================================
step_header(3, "Data Preprocessing", "Drop missing price → clean numerics → mode-impute superhost → type fixes", color="#f59e0b")

st.markdown("""
**Pointers**  
- We **do not** impute the target (`price`): rows with missing `price` are dropped.  
- Key numerics are **coerced to numeric** and rows with missing numerics are dropped — this ensures only **`host_is_superhost`** can remain missing.  
- Then we **mode-impute** `host_is_superhost` and fix boolean/categorical types.
""")

# 3.1 — Drop rows with missing price (no target imputation)
df[TARGET] = clean_price_series(df[TARGET])
before_drop_price = len(df)
df = df.dropna(subset=[TARGET]).copy()
st.info(f"🧹 Dropped **{before_drop_price - len(df)}** rows with missing `price`; kept **{len(df)}** rows.")

# 3.2 — Coerce numerics & drop rows with missing numerics so only 'host_is_superhost' remains missing
core_numeric = [c for c in ["bathrooms", "bedrooms", "number_of_reviews", "latitude", "longitude"] if c in df.columns]
for c in core_numeric:
    df[c] = pd.to_numeric(df[c], errors="coerce")

before_drop_num = len(df)
df = df.dropna(subset=core_numeric).copy()
after_drop_num = len(df)
if before_drop_num != after_drop_num:
    st.caption(f"Ensured clean numerics: dropped **{before_drop_num - after_drop_num}** rows with missing {core_numeric}")

# 3.3 — Show missingness *now* (after price drop + numeric cleanup)
st.subheader("Missingness after price-drop & numeric cleanup")
miss_now = (df.isna().mean() * 100).round(2).to_frame("missing_%")
st.dataframe(
    miss_now.style.format({"missing_%": "{:.2f}"})
            .background_gradient(cmap="Greens", axis=None),
    use_container_width=True
)

st.caption("Expected: only `host_is_superhost` should remain missing at this stage.")

# 3.4 — Mode imputation for host_is_superhost
if "host_is_superhost" in df.columns:
    mode_val = df["host_is_superhost"].mode(dropna=True)
    if not mode_val.empty:
        fill_value = mode_val.iloc[0]
        df["host_is_superhost"] = df["host_is_superhost"].fillna(fill_value)
        st.success(f"Filled missing `host_is_superhost` with **mode = {fill_value!r}**")
    else:
        st.warning("Could not compute a mode for `host_is_superhost` (no non-null values).")
else:
    st.info("`host_is_superhost` not present in the dataset.")

# 3.5 — Type fixes
for c in ["host_identity_verified", "host_is_superhost"]:
    if c in df.columns:
        df[c] = to_bool_t_f(df[c])
if "room_type" in df.columns:
    df["room_type"] = df["room_type"].astype("category")

# 3.6 — Missingness after imputation (should be near 0%)
st.subheader("Missingness after imputation & type fixes")
miss_final = (df.isna().mean() * 100).round(2).to_frame("missing_%")
st.dataframe(
    miss_final.style.format({"missing_%": "{:.2f}"})
             .background_gradient(cmap="Blues", axis=None),
    use_container_width=True
)
st.markdown('<hr class="soft"/>', unsafe_allow_html=True)

# ====================================================
# Section 4 — Exploratory Data Analysis (EDA)
# ====================================================
step_header(4, "Exploratory Data Analysis (EDA)", "Describe stats • Distributions • Correlations • Outliers • Map", color="#8b5cf6")

# 4.1: Descriptive statistics
st.subheader("4.1 Descriptive Statistics (Numeric)")
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if numeric_cols:
    st.dataframe(
        df[numeric_cols].describe().T.round(3)
          .style.set_properties(**{"text-align":"left"})
          .set_table_styles([{"selector":"th","props":[("text-align","left"),("background","#0b1220"),("color","white"),("padding","6px 8px")]}]),
        use_container_width=True
    )
else:
    st.info("No numeric columns available for describe().")

# 4.2: Compact side-by-side distributions (Numeric vs Categorical)
st.subheader("4.2 Distributions — Side by Side")
num_plot_cols = [c for c in ["price", "bedrooms", "bathrooms", "number_of_reviews"] if c in df.columns]
cat_plot_cols = [c for c in ["room_type", "host_identity_verified", "host_is_superhost"] if c in df.columns]

col_num, col_cat = st.columns(2, gap="large")
with col_num:
    st.markdown("**Numeric**")
    if num_plot_cols:
        for c in num_plot_cols:
            fig, ax = plt.subplots(figsize=(4.8, 3.0))
            sns.histplot(df[c].dropna(), bins=36, ax=ax, kde=False)
            ax.set_xlabel(c); ax.set_ylabel("")
            ax.grid(axis="y", linestyle=":", alpha=0.35)
            st.pyplot(fig)
    else:
        st.info("No numeric features available.")

with col_cat:
    st.markdown("**Categorical**")
    if cat_plot_cols:
        for c in cat_plot_cols:
            vc = df[c].astype("object").value_counts(dropna=False).head(20)
            fig, ax = plt.subplots(figsize=(4.8, 3.0))
            sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax)
            ax.set_xlabel("Count"); ax.set_ylabel("")
            for i, v in enumerate(vc.values):
                ax.text(v, i, f" {v}", va="center")
            ax.grid(axis="x", linestyle=":", alpha=0.35)
            st.pyplot(fig)
    else:
        st.info("No categorical features available.")

# 4.3: Pair Plot (sampled for performance)
st.subheader("4.3 Pair Plot")
pp_cols = [c for c in ["price", "bedrooms", "bathrooms", "number_of_reviews"] if c in df.columns]
if len(pp_cols) >= 2 and len(df) > 1:
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
    st.caption(f"Applying cap at 99th percentile: price ≤ {cap:,.2f}")
    df_wo = df[df["price"] <= cap].copy()

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.histplot(df["price"], bins=40, ax=ax, color="#ef4444")
        ax.set_title("Before outlier handling"); ax.set_xlabel("price"); ax.set_ylabel("count")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        sns.histplot(df_wo["price"], bins=40, ax=ax, color="#10b981")
        ax.set_title("After outlier handling (≤ 99th pct)"); ax.set_xlabel("price"); ax.set_ylabel("count")
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
        height=520,
        title="Asheville — Listings by price bucket"
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Latitude/Longitude/Price not all available for a geospatial plot.")

st.markdown('<hr class="soft"/>', unsafe_allow_html=True)

# ====================================================
# Section 5 — Prepare for Modeling (Normalize & Split) + Keras Model
# ====================================================
step_header(5, "Modeling (Normalize, Split & Train)", "Keras Sequential with Normalization; train/valid curves", color="#ef4444")

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

model.compile(loss="mse", optimizer="adam", metrics=["mae"])

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
step_header(6, "Final Results", "MAE • RMSE • R² — interpret performance & next steps", color="#334155")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))
r2 = r2_score(y_test, y_pred)

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:,.2f}")
m2.metric("RMSE", f"{rmse:,.2f}")
m3.metric("R²", f"{r2:.3f}")

st.caption("Notes: MAE/RMSE reflect average error; R² measures explained variance. Training curves show convergence; the map highlights geographic pricing patterns.")
