# app.py — Airbnb Price Prediction (Detailed Workflow, Graphs First → Results Last)
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Airbnb Price Prediction", page_icon="🏠", layout="wide")
st.title("🏠 Airbnb Price Prediction — Detailed End-to-End Workflow")

# ----------------------------
# Helpers
# ----------------------------
DATA_PATH = Path("listings.csv")  # Hardcoded path
TARGET = "price"

def clean_price_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def split_num_cat(df: pd.DataFrame, exclude=None):
    exclude = set(exclude or [])
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in exclude]
    return num_cols, cat_cols

def eval_regression(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # version-safe RMSE
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ====================================================
# Section 1 — Data Loading (What this section does)
# - Read the Inside Airbnb dataset (listings.csv)
# - Show shape + a quick preview to verify columns
# ====================================================
st.header("Section 1 — Data Loading")

if not DATA_PATH.exists():
    st.error("❌ File not found: `listings.csv`. Please add the dataset and rerun.")
    st.stop()

df = pd.read_csv(DATA_PATH, low_memory=False)
st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns from `{DATA_PATH}`")
st.caption("First 5 rows (sanity check of schema and sample values):")
st.dataframe(df.head(), use_container_width=True)

# ====================================================
# Section 2 — Data Selection (What this section does)
# - Fix target = `price` (clean $ and commas)
# - Choose a stable, portable set of predictive features
# - Keep only target + selected features for the workflow
# ====================================================
st.header("Section 2 — Data Selection")

if TARGET not in df.columns:
    st.error("Column `price` not found in the dataset.")
    st.stop()

df[TARGET] = clean_price_series(df[TARGET])

candidate_defaults = [
    "accommodates", "bedrooms", "bathrooms", "beds",
    "number_of_reviews", "review_scores_rating",
    "minimum_nights", "availability_365",
    "latitude", "longitude",
    "room_type", "neighbourhood", "neighbourhood_cleansed", "property_type"
]
features = [c for c in candidate_defaults if c in df.columns]
if not features:
    # Fallback if city export is unusual—cap to avoid huge wide models
    features = [c for c in df.columns if c != TARGET][:20]

st.markdown("**Selected Features:** " + ", ".join(features))
work = df[features + [TARGET]].copy()

# ====================================================
# Section 3 — Data Preprocessing (What this section does)
# - Numeric: coerce to numeric, impute missing with mean
# - Categorical: cast to object, impute missing with 'Unknown'
# - Drop remaining NA rows in focused dataset
# - Show a quick missingness report (original selection)
# ====================================================
st.header("Section 3 — Data Preprocessing")

num_cols, cat_cols = split_num_cat(work.drop(columns=[TARGET]))
# Impute numeric
for c in num_cols:
    work[c] = pd.to_numeric(work[c], errors="coerce").fillna(work[c].mean())
# Impute categorical
for c in cat_cols:
    work[c] = work[c].astype(object).fillna("Unknown")

before = len(work)
work = work.dropna(subset=features + [TARGET])
after = len(work)
st.info(f"Cleaned dataset has **{after:,} rows** (removed **{before - after:,}** incomplete rows).")

st.caption("Original missingness (before imputation) within selected columns:")
miss_tbl = df[features + [TARGET]].isna().mean().to_frame("missing_ratio")
st.dataframe((miss_tbl * 100).round(2), use_container_width=True)

# ====================================================
# Section 4 — EDA (What this section does)
# - Visualize target distribution (price)
# - Visualize numeric correlations (heatmap)
# - Optional: simple bivariate charts if present (room_type, property_type)
# ====================================================
st.header("Section 4 — Exploratory Data Analysis (EDA)")

# 4.1 Target distribution
c1, c2 = st.columns(2)
with c1:
    st.subheader("Distribution of Price")
    fig, ax = plt.subplots()
    sns.histplot(work[TARGET], bins=40, ax=ax, color="#3b82f6")
    ax.set_xlabel("Price"); ax.set_ylabel("Count")
    st.pyplot(fig)

# 4.2 Correlation heatmap (numeric)
with c2:
    st.subheader("Correlation Heatmap (Numeric Features)")
    corr_cols = [c for c in num_cols if c in work.columns] + [TARGET]
    if len(corr_cols) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(work[corr_cols].corr(numeric_only=True), cmap="YlGnBu", annot=False, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")

# 4.3 Optional categorical views
cat_view_cols = [c for c in ["room_type", "property_type", "neighbourhood"] if c in work.columns]
if cat_view_cols:
    st.subheader("Category Counts (Top Levels)")
    for c in cat_view_cols:
        vc = work[c].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(x=vc.values, y=vc.index, ax=ax, color="#10b981")
        ax.set_xlabel("Count"); ax.set_ylabel(c)
        st.pyplot(fig)

# ====================================================
# Section 5 — Modeling (What this section does)
# - Normalize numeric features; One-hot encode categoricals
# - Split into Train/Test (80/20, seed=42)
# - Train Linear Regression baseline
# - Show **graphs first**: coefficients (importance proxy) & Predicted vs Actual
# - Finally, display metrics (MAE, RMSE, R²)
# ====================================================
st.header("Section 5 — Modeling")

X = work[features].copy()
y = work[TARGET].astype(float)

num_feats, cat_feats = split_num_cat(X)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ],
    remainder="drop"
)

pipe = Pipeline([
    ("prep", preprocess),
    ("model", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
pipe.fit(X_train, y_train)

# ---- Graphs first ----

# 5.A Standardized coefficients (importance proxy)
st.subheader("Feature Influence (Standardized Coefficients)")
try:
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(cat_feats) if cat_feats else []
    feature_names = np.array(num_feats + list(cat_names))

    coefs = pipe.named_steps["model"].coef_
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
    coef_df["Abs"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35*len(coef_df))))
    sns.barplot(x="Coefficient", y="Feature", data=coef_df, ax=ax, palette="viridis")
    ax.set_xlabel("Standardized Coefficient"); ax.set_ylabel("")
    st.pyplot(fig)
except Exception:
    st.info("Could not compute/display coefficients (e.g., all features categorical).")

# 5.B Predicted vs Actual (Test)
st.subheader("Predicted vs Actual (Test Set)")
y_pred = pipe.predict(X_test)
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.45, color="#0ea5e9")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual Price"); ax.set_ylabel("Predicted Price")
st.pyplot(fig)

# ---- Final Results (after graphs) ----
mae, rmse, r2 = eval_regression(y_test, y_pred)

st.subheader("Final Results")
r1, r2col, r3 = st.columns(3)
r1.metric("MAE", f"{mae:,.2f}")
r2col.metric("RMSE", f"{rmse:,.2f}")
r3.metric("R²", f"{r2:.3f}")

st.caption("Notes: MAE/RMSE reflect average error; R² measures explained variance. Coefficients indicate relative influence after scaling & encoding.")
