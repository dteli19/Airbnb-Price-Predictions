# app.py — Airbnb Price Prediction (EDA → Preprocessing → Modeling)
# Streamlit app with Sections 1–5:
# 1) Data Loading  2) Data Selection  3) Data Preprocessing
# 4) EDA  5) Prepare for Modeling (Normalize & Split) + Regression

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Page / Style
# ----------------------------
st.set_page_config(page_title="Airbnb Price Prediction", page_icon="🏠", layout="wide")
st.title("🏠 Airbnb Price Prediction — EDA → Preprocessing → Modeling")

# ----------------------------
# Hero copy (Portfolio style)
# ----------------------------
with st.expander("Overview • Problem • Data • Actions • Results", expanded=True):
    st.markdown("""
**Overview**  
This app demonstrates an end-to-end workflow for **Airbnb price prediction** using an open Inside Airbnb dataset. It walks through data loading, feature selection, preprocessing, exploratory analysis, and a baseline regression model.

**Problem Statement**  
Given listing attributes (room type, location, reviews, capacity, etc.), predict the **nightly price**. The goal is to build an interpretable baseline that surfaces the most influential features and establishes metrics for iteration.

**About the Data**  
We work with a standard Inside Airbnb export (CSV) containing columns like:  
`price`, `accommodates`, `bedrooms`, `bathrooms`, `number_of_reviews`, `review_scores_rating`, `latitude`, `longitude`, `room_type`, `neighbourhood`, `minimum_nights`, `availability_365`, etc.  
(Columns vary by city; the app adapts to your file.)

**Actions (what this app does)**  
1) **Load** a listings CSV.  
2) **Select** target and candidate features.  
3) **Preprocess**: clean `price`, handle missing values, encode categoricals, normalize numerics.  
4) **EDA**: schema, distributions, correlations.  
5) **Model**: train/test split, baseline **Linear Regression** (optionally Ridge), report **MAE, RMSE, R²**, and show coefficients.

**Results**  
You’ll get an interpretable baseline with evaluation metrics, standardized coefficients (importance proxy), and plots to understand the data and model behavior.
""")

# ----------------------------
# Utilities
# ----------------------------
def clean_price_series(s: pd.Series) -> pd.Series:
    """Strip $, commas, spaces; coerce to float."""
    if s.dtype == object:
        s = s.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def numeric_categorical_split(df: pd.DataFrame, exclude=None):
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
# Section 1 — Data Loading
# ----------------------------
st.header("1) Data Loading")
left, right = st.columns([1.2, 1])
with left:
    uploaded = st.file_uploader("Upload Inside Airbnb listings CSV", type=["csv"])
    st.caption("Tip: use the `listings.csv` from https://insideairbnb.com (download locally, then upload here).")

df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded, low_memory=False)
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, engine="python", low_memory=False)

if df is None:
    st.info("Upload a CSV to continue.")
    st.stop()

st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns")
with right:
    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

# ----------------------------
# Section 2 — Data Selection
# ----------------------------
st.header("2) Data Selection")
default_target = "price" if "price" in df.columns else None
target = st.selectbox("Target (dependent variable)", options=[default_target] + sorted([c for c in df.columns if c != default_target]) if default_target else sorted(df.columns))
candidate_feats = [c for c in df.columns if c != target]

# Suggest commonly useful features if present
suggested = [c for c in [
    "accommodates", "bedrooms", "bathrooms", "beds",
    "number_of_reviews", "review_scores_rating", "minimum_nights",
    "availability_365", "latitude", "longitude",
    "room_type", "neighbourhood_cleansed", "neighbourhood", "property_type"
] if c in candidate_feats]

features = st.multiselect("Candidate features", options=candidate_feats, default=suggested or candidate_feats[:12])
if not features:
    st.warning("Select at least one feature.")
    st.stop()

# ----------------------------
# Section 3 — Data Preprocessing
# ----------------------------
st.header("3) Data Preprocessing")

work = df[features + [target]].copy()

# Clean target price if needed
if target.lower() == "price":
    work[target] = clean_price_series(work[target])

# Basic missingness report
miss_tbl = work.isna().mean().sort_values(ascending=False).to_frame("missing_ratio")
st.subheader("Missingness")
st.dataframe((miss_tbl * 100).round(2), use_container_width=True)

# Simple imputation strategy slider
st.caption("Missing value handling:")
impute_num = st.selectbox("Numeric missing values", ["Drop rows", "Fill with column mean"], index=1)
impute_cat = st.selectbox("Categorical missing values", ["Drop rows", "Fill with 'Unknown'"], index=1)

# Apply light imputation for EDA/modeling convenience
num_cols, cat_cols = numeric_categorical_split(work.drop(columns=[target]))
if impute_num == "Fill with column mean":
    for c in num_cols:
        work[c] = work[c].fillna(work[c].mean())
if impute_cat == "Fill with 'Unknown'":
    for c in cat_cols:
        work[c] = work[c].fillna("Unknown")

# Drop rows that still have NA in selected columns/target
before = len(work)
work = work.dropna(subset=features + [target])
after = len(work)
st.caption(f"Dropped {before - after} rows after preprocessing (kept {after}).")

# ----------------------------
# Section 4 — EDA
# ----------------------------
st.header("4) Exploratory Data Analysis (EDA)")
eda1, eda2 = st.columns(2)

with eda1:
    st.subheader("Numeric Distributions")
    if num_cols:
        sel_num = st.selectbox("Numeric feature", options=num_cols)
        fig, ax = plt.subplots()
        sns.histplot(work[sel_num], bins=40, ax=ax)
        ax.set_xlabel(sel_num)
        st.pyplot(fig)
    else:
        st.info("No numeric features selected.")

with eda2:
    st.subheader("Categorical Counts")
    if cat_cols:
        sel_cat = st.selectbox("Categorical feature", options=cat_cols)
        vc = work[sel_cat].value_counts().head(20)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=vc.values, y=vc.index, ax=ax)
        ax.set_xlabel("count"); ax.set_ylabel(sel_cat)
        st.pyplot(fig)
    else:
        st.info("No categorical features selected.")

st.subheader("Correlation (numeric features)")
if len(num_cols) >= 2:
    corr = work[num_cols + ([target] if pd.api.types.is_numeric_dtype(work[target]) else [])].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, cmap="Blues", annot=False, ax=ax)
    st.pyplot(fig)
else:
    st.info("Not enough numeric columns for a correlation heatmap.")

# ----------------------------
# Section 5 — Prepare for Modeling & Train
# ----------------------------
st.header("5) Prepare for Modeling & Evaluation")

# Target / feature arrays
y = work[target].astype(float) if target in work.columns else None
X = work[features].copy()

# Split controls
c1, c2, c3 = st.columns(3)
with c1:
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
with c2:
    random_state = st.number_input("Random state", value=42, step=1)
with c3:
    model_name = st.selectbox("Model", ["Linear Regression", "Ridge (α=1.0)"], index=0)

# Column transformer: scale numerics, one-hot categoricals
numeric_features, categorical_features = numeric_categorical_split(X)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="drop"
)

if model_name == "Linear Regression":
    model = LinearRegression()
else:
    model = Ridge(alpha=1.0, random_state=None)

pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])

# Train/test split + fit
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)

pipe.fit(X_train, y_train)
pred_train = pipe.predict(X_train)
pred_test  = pipe.predict(X_test)

mae_tr, rmse_tr, r2_tr = eval_regression(y_train, pred_train)
mae_te, rmse_te, r2_te = eval_regression(y_test, pred_test)

st.subheader("Evaluation Metrics")
m1, m2, m3 = st.columns(3)
m1.metric("Test MAE", f"{mae_te:,.2f}")
m2.metric("Test RMSE", f"{rmse_te:,.2f}")
m3.metric("Test R²", f"{r2_te:,.3f}")

st.caption(f"Train → MAE: {mae_tr:,.2f} | RMSE: {rmse_tr:,.2f} | R²: {r2_tr:,.3f}")

# Coefficients (approximate importance for linear models)
st.subheader("Feature Effects (standardized coefficients)")
try:
    # Get feature names post-encoding
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(categorical_features) if categorical_features else np.array([])
    feature_names = np.array(numeric_features + list(cat_names))

    if hasattr(pipe.named_steps["model"], "coef_"):
        coefs = pipe.named_steps["model"].coef_
        coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
        coef_df["abs_coef"] = coef_df["coef"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(7, max(3.5, 0.35 * len(coef_df))))
        sns.barplot(x="coef", y="feature", data=coef_df, ax=ax)
        ax.set_xlabel("Standardized coefficient")
        ax.set_ylabel("")
        st.pyplot(fig)
    else:
        st.info("This model does not expose linear coefficients.")
except Exception as e:
    st.info("Could not compute coefficient plot (possibly due to all-categorical input).")

# Scatter: y_true vs y_pred (test)
st.subheader("Predicted vs Actual (Test Set)")
fig, ax = plt.subplots()
ax.scatter(y_test, pred_test, alpha=0.4)
lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual price"); ax.set_ylabel("Predicted price")
st.pyplot(fig)

# Download predictions
out = pd.DataFrame({
    "y_test": y_test.reset_index(drop=True),
    "y_pred": pd.Series(pred_test)
})
st.download_button("⬇️ Download predictions (CSV)",
                   data=out.to_csv(index=False),
                   file_name="airbnb_price_predictions.csv",
                   mime="text/csv")

st.caption("Note: Results depend on the city/file you upload and the features you select. Use this as a baseline to iterate with regularization, feature engineering, or tree models.")
