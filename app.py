# app.py — Airbnb Price Prediction (static Colab-style)
# Sections: Overview, Problem, About Data, Actions, Results
# 1) Data Loading  2) Data Selection  3) Data Preprocessing
# 4) EDA  5) Prepare for Modeling (Normalize & Split) + Linear Regression

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
# Page / Style
# ----------------------------
st.set_page_config(page_title="Airbnb Price Prediction", page_icon="🏠", layout="wide")
st.title("🏠 Airbnb Price Prediction — Colab-Style Static Pipeline")

# ----------------------------
# Overview • Problem • Data • Actions • Results
# ----------------------------
with st.expander("Overview • Problem • Data • Actions • Results", expanded=True):
    st.markdown("""
**Overview**  
End-to-end workflow for **Airbnb price prediction** using an Inside Airbnb dataset. Static steps mirror a typical Colab notebook: load → clean → EDA → preprocess → model.

**Problem Statement**  
Given listing attributes (capacity, reviews, location, room type), predict the **nightly price** and surface influential features.

**About the Data**  
Inside Airbnb `listings.csv` with columns such as: `price`, `accommodates`, `bedrooms`, `bathrooms`, `number_of_reviews`, `review_scores_rating`, `latitude`, `longitude`, `room_type`, `neighbourhood`, `minimum_nights`, `availability_365`, etc. Columns vary by city.

**Actions**  
- Clean `price`, select common features  
- Fixed missing-data handling (**numeric → mean**, **categorical → 'Unknown'**)  
- Normalize numerics, one-hot encode categoricals  
- Train/test split (80/20, seed=42), **Linear Regression** baseline  
- Report **MAE, RMSE, R²** + standardized coefficients

**Results**  
Interpretable baseline metrics and top standardized coefficients to guide iteration.
""")

# ----------------------------
# Utilities
# ----------------------------
DATA_PATH = Path("data/listings.csv")  # <- hardcoded
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
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ----------------------------
# 1) Data Loading (static)
# ----------------------------
st.header("1) Data Loading")
if not DATA_PATH.exists():
    st.error("`data/listings.csv` not found. Place your Inside Airbnb file at this path and rerun.")
    st.stop()

# robust CSV read
try:
    df = pd.read_csv(DATA_PATH, low_memory=False)
except Exception:
    df = pd.read_csv(DATA_PATH, engine="python", low_memory=False)

st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns from `{DATA_PATH}`")
st.dataframe(df.head(), use_container_width=True)

# ----------------------------
# 2) Data Selection (static defaults, intersect with available columns)
# ----------------------------
st.header("2) Data Selection")

# Clean price if present
if TARGET in df.columns:
    df[TARGET] = clean_price_series(df[TARGET])
else:
    st.error("Column `price` not found in listings.csv. Ensure the Inside Airbnb export includes `price`.")
    st.stop()

# Common, Colab-like candidate features (will be intersected with actual columns)
candidate_defaults = [
    "accommodates", "bedrooms", "bathrooms", "beds",
    "number_of_reviews", "review_scores_rating",
    "minimum_nights", "availability_365",
    "latitude", "longitude",
    "room_type", "neighbourhood_cleansed", "neighbourhood", "property_type"
]
features = [c for c in candidate_defaults if c in df.columns]
if not features:
    # Fallback: use all non-target columns (capped to 20 for stability)
    features = [c for c in df.columns if c != TARGET][:20]

st.caption("Using fixed feature set (Colab-style, no UI choices).")
st.write("**Features used:**", features)

work = df[features + [TARGET]].copy()

# ----------------------------
# 3) Data Preprocessing (static rules)
# ----------------------------
st.header("3) Data Preprocessing (static)")

# Fixed missing-data handling
# Numeric → mean; Categorical → 'Unknown'; then drop remaining NA in target/features
num_cols, cat_cols = split_num_cat(work.drop(columns=[TARGET]))
for c in num_cols:
    work[c] = pd.to_numeric(work[c], errors="coerce").fillna(work[c].mean())
for c in cat_cols:
    work[c] = work[c].astype(object).fillna("Unknown")

before = len(work)
work = work.dropna(subset=features + [TARGET])
after = len(work)
st.caption(f"Dropped {before - after} rows after preprocessing (kept {after}).")

# Missingness table (post-imputation/pre-drop)
miss_tbl = df[features + [TARGET]].isna().mean().sort_values(ascending=False).to_frame("missing_ratio")
st.subheader("Missingness (original selection)")
st.dataframe((miss_tbl * 100).round(2), use_container_width=True)

# ----------------------------
# 4) Exploratory Data Analysis (EDA)
# ----------------------------
st.header("4) Exploratory Data Analysis (EDA)")
eda1, eda2 = st.columns(2)

with eda1:
    st.subheader("Target Distribution (price)")
    fig, ax = plt.subplots()
    sns.histplot(work[TARGET], bins=50, ax=ax)
    ax.set_xlabel("price")
    st.pyplot(fig)

with eda2:
    st.subheader("Correlation (numeric only)")
    corr_num = [c for c in num_cols if c in work.columns] + [TARGET]
    corr_num = [c for c in corr_num if pd.api.types.is_numeric_dtype(work[c])]
    if len(corr_num) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(work[corr_num].corr(numeric_only=True), cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")

# ----------------------------
# 5) Prepare for Modeling (Normalize & Split) + Linear Regression
# ----------------------------
st.header("5) Prepare for Modeling & Train (static)")

X = work[features].copy()
y = work[TARGET].astype(float)

numeric_features, categorical_features = split_num_cat(X)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="drop"
)

model = LinearRegression()
pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])

# Fixed split (Colab style): 80/20, seed=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

pipe.fit(X_train, y_train)
pred_train = pipe.predict(X_train)
pred_test = pipe.predict(X_test)

mae_tr, rmse_tr, r2_tr = eval_regression(y_train, pred_train)
mae_te, rmse_te, r2_te = eval_regression(y_test, pred_test)

st.subheader("Evaluation Metrics (Test)")
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae_te:,.2f}")
m2.metric("RMSE", f"{rmse_te:,.2f}")
m3.metric("R²", f"{r2_te:,.3f}")
st.caption(f"Train → MAE: {mae_tr:,.2f} | RMSE: {rmse_tr:,.2f} | R²: {r2_tr:,.3f}")

# Coefficient importance (standardized)
st.subheader("Top Standardized Coefficients (Linear Model)")
try:
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(categorical_features) if categorical_features else np.array([])
    feature_names = np.array(numeric_features + list(cat_names))

    coefs = pipe.named_steps["model"].coef_
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.35 * len(coef_df))))
    sns.barplot(x="coef", y="feature", data=coef_df, ax=ax)
    ax.set_xlabel("Standardized coefficient")
    ax.set_ylabel("")
    st.pyplot(fig)
except Exception:
    st.info("Could not render coefficients (e.g., all-categorical features).")

# Predicted vs Actual
st.subheader("Predicted vs Actual (Test)")
fig, ax = plt.subplots()
ax.scatter(y_test, pred_test, alpha=0.4)
lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual price"); ax.set_ylabel("Predicted price")
st.pyplot(fig)

# Download predictions
out = pd.DataFrame({"y_test": y_test.reset_index(drop=True), "y_pred": pd.Series(pred_test)})
st.download_button(
    "⬇️ Download predictions (CSV)",
    data=out.to_csv(index=False),
    file_name="airbnb_price_predictions.csv",
    mime="text/csv",
)
