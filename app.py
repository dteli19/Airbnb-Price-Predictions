# app.py — Airbnb Price Prediction (static Colab-style + guidance & observations)
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
# Page config
# ----------------------------
st.set_page_config(page_title="Airbnb Price Prediction", page_icon="🏠", layout="wide")
st.title("🏠 Airbnb Price Prediction — End-to-End (Colab-style)")

# ----------------------------
# Helpers
# ----------------------------
DATA_PATH = Path("listings.csv")  # hardcoded
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
    mse = mean_squared_error(y_true, y_pred)  # version-safe
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def price_quick_stats(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
        "skew": float(s.skew()) if s.count() > 2 else 0.0,
    }

# ====================================================
# Section 1 — Overview, Problem Statement, About the Data
# ====================================================
st.header("Section 1 — Overview, Problem Statement, About the Data")

st.markdown("""
**Overview**  
This project predicts **Airbnb nightly prices** using an Inside Airbnb dataset. It mirrors a Colab workflow: load data, clean/prepare, explore, and build a baseline regression model.

**Problem Statement**  
Given listing attributes (capacity, reviews, location, room/prop types), estimate the **price**, and surface which features influence price most in a simple, interpretable baseline.

**About the Data**  
Inside Airbnb `listings.csv` typically includes: `price`, `accommodates`, `bedrooms`, `bathrooms`, `number_of_reviews`, `review_scores_rating`, `minimum_nights`, `availability_365`, `latitude`, `longitude`, `room_type`, `neighbourhood`, `neighbourhood_cleansed`, `property_type`, etc.  
(Columns vary by city; this pipeline adapts to what’s available.)
""")

with st.expander("What to look for (Section 1)"):
    st.markdown("""
- Confirm that the dataset contains `price` and a reasonable set of predictors.  
- Check if the city export has `room_type`, `accommodates`, and review fields—these usually carry strong signal.  
""")

# ====================================================
# Section 2 — Actions / Steps to be Performed
# ====================================================
st.header("Section 2 — Actions / Steps to be Performed")
st.markdown("""
1) **Data Loading**: Read `data/listings.csv`.  
2) **Data Selection**: Use a fixed feature set (intersect with available columns); target = `price`.  
3) **Data Preprocessing**: Clean `price`; impute numeric with **mean**; categorical with **'Unknown'**; drop residual NA.  
4) **EDA**: Inspect target distribution and numeric correlation heatmap.  
5) **Prepare for Modeling**: Normalize numerics, one-hot encode categoricals; split 80/20 (seed=42); **Linear Regression** baseline; report **MAE, RMSE, R²**; show standardized coefficients & Predicted vs Actual.
""")

with st.expander("What we did (Section 2)"):
    st.markdown("""
- Kept preprocessing **fixed** (no user choices) to match Colab.  
- Ensured reproducibility with a fixed **random_state=42** for the split.  
""")

# ====================================================
# Section 3 — Notebook Workflow (1→5)
# ====================================================
st.header("Section 3 — Notebook Workflow")

# 3.1 Data Loading
st.subheader("3.1 Data Loading")
if not DATA_PATH.exists():
    st.error("File not found: `data/listings.csv`. Please add the Inside Airbnb CSV there and rerun.")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH, low_memory=False)
except Exception:
    df = pd.read_csv(DATA_PATH, engine="python", low_memory=False)

st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns from `{DATA_PATH}`")
st.dataframe(df.head(), use_container_width=True)

with st.expander("What to look for (Data Loading)"):
    st.markdown("""
- Data shape and a quick glance of the first rows: do key columns exist and look clean (no all-null columns)?  
""")

# 3.2 Data Selection (fixed)
st.subheader("3.2 Data Selection")
if TARGET not in df.columns:
    st.error("Column `price` not found in listings.csv.")
    st.stop()

df[TARGET] = clean_price_series(df[TARGET])

candidate_defaults = [
    "accommodates", "bedrooms", "bathrooms", "beds",
    "number_of_reviews", "review_scores_rating",
    "minimum_nights", "availability_365",
    "latitude", "longitude",
    "room_type", "neighbourhood_cleansed", "neighbourhood", "property_type"
]
features = [c for c in candidate_defaults if c in df.columns]
if not features:
    # Fallback: safe cap if city export is unusual
    features = [c for c in df.columns if c != TARGET][:20]

st.caption("Using a fixed feature set (Colab-style).")
st.write("**Features used:**", features)

work = df[features + [TARGET]].copy()

with st.expander("What we did (Data Selection)"):
    st.markdown("""
- Chose a **standard set** of Airbnb predictors often present across cities.  
- Kept target fixed as **`price`** and took the **intersection** with actual columns.  
""")

# 3.3 Data Preprocessing (fixed)
st.subheader("3.3 Data Preprocessing (fixed rules)")

num_cols, cat_cols = split_num_cat(work.drop(columns=[TARGET]))
# Numeric → mean; Categorical → 'Unknown'
for c in num_cols:
    work[c] = pd.to_numeric(work[c], errors="coerce").fillna(work[c].mean())
for c in cat_cols:
    work[c] = work[c].astype(object).fillna("Unknown")

before = len(work)
work = work.dropna(subset=features + [TARGET])
after = len(work)
st.caption(f"Dropped {before - after} rows after preprocessing (kept {after}).")

# Missingness (original selection)
miss_tbl = df[features + [TARGET]].isna().mean().sort_values(ascending=False).to_frame("missing_ratio")
st.write("**Missingness (original selection, %)**")
st.dataframe((miss_tbl * 100).round(2), use_container_width=True)

with st.expander("What to look for (Preprocessing)"):
    st.markdown("""
- Are there columns with **very high missingness** (e.g., >40%) that might be noisy?  
- After imputation, did we **retain enough rows** for a reliable split?  
""")

# 3.4 EDA
st.subheader("3.4 Exploratory Data Analysis (EDA)")
c1, c2 = st.columns(2)

with c1:
    st.write("**Target Distribution (price)**")
    fig, ax = plt.subplots()
    sns.histplot(work[TARGET], bins=50, ax=ax)
    ax.set_xlabel("price")
    st.pyplot(fig)

with c2:
    st.write("**Correlation (numeric only)**")
    corr_num = [c for c in num_cols if c in work.columns] + [TARGET]
    corr_num = [c for c in corr_num if pd.api.types.is_numeric_dtype(work[c])]
    if len(corr_num) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(work[corr_num].corr(numeric_only=True), cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")

# Auto-observations (EDA)
auto_obs = []
# Price stats
stats = price_quick_stats(work[TARGET])
if stats:
    auto_obs.append(f"- **Price stats:** mean ≈ {stats['mean']:.2f}, median ≈ {stats['median']:.2f}, std ≈ {stats['std']:.2f}; range [{stats['min']:.2f}, {stats['max']:.2f}].")
    if abs(stats["skew"]) > 1:
        auto_obs.append("- **Skewness:** price appears **highly skewed**; consider log-transform in later iterations.")
    elif abs(stats["skew"]) > 0.5:
        auto_obs.append("- **Skewness:** price shows **moderate skew**.")
# Top numeric correlations with price
if len(corr_num) >= 2:
    corrs = work[corr_num].corr(numeric_only=True)[TARGET].drop(TARGET).dropna()
    if not corrs.empty:
        top_corr = corrs.abs().sort_values(ascending=False).head(3).index.tolist()
        auto_obs.append(f"- **Top numeric correlates with price:** {', '.join(top_corr)}.")

with st.expander("Observations (from this EDA)"):
    if auto_obs:
        st.markdown("\n".join(auto_obs))
    else:
        st.markdown("- No strong numeric correlations detected or insufficient numeric features.")

# 3.5 Prepare for Modeling (Normalize & Split) + Linear Regression
st.subheader("3.5 Prepare for Modeling (Normalize & Split Data) + Train")

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

pipe = Pipeline(steps=[("prep", preprocess), ("model", LinearRegression())])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

pipe.fit(X_train, y_train)
pred_train = pipe.predict(X_train)
pred_test  = pipe.predict(X_test)

mae_tr, rmse_tr, r2_tr = eval_regression(y_train, pred_train)
mae_te, rmse_te, r2_te = eval_regression(y_test, pred_test)

with st.expander("What we did (Model Prep & Train)"):
    st.markdown("""
- Standardized **numeric** features; one-hot encoded **categoricals**.  
- Fixed **80/20 split** (seed=42).  
- Trained **Linear Regression** for a simple, interpretable baseline.  
""")

# ====================================================
# Section 4 — Results
# ====================================================
st.header("Section 4 — Results")

m1, m2, m3 = st.columns(3)
m1.metric("Test MAE", f"{mae_te:,.2f}")
m2.metric("Test RMSE", f"{rmse_te:,.2f}")
m3.metric("Test R²", f"{r2_te:,.3f}")
st.caption(f"Train → MAE: {mae_tr:,.2f} | RMSE: {rmse_tr:,.2f} | R²: {r2_tr:,.3f}")

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
    st.info("Could not render coefficients (e.g., all-categorical inputs).")

st.subheader("Predicted vs Actual (Test)")
fig, ax = plt.subplots()
ax.scatter(y_test, pred_test, alpha=0.4)
lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual price"); ax.set_ylabel("Predicted price")
st.pyplot(fig)

with st.expander("What to look for (Results)"):
    st.markdown("""
- **Generalization gap:** Compare Train vs Test metrics—overfitting if Train ≪ Test errors.  
- **Feature effects:** Which standardized coefficients are largest? Do they align with domain expectations (e.g., `accommodates`, `room_type`)?  
- **Prediction quality:** Points tightly along the 45° line indicate better fit.  
""")

# Download predictions (for inspection)
out = pd.DataFrame({"y_test": y_test.reset_index(drop=True), "y_pred": pd.Series(pred_test)})
st.download_button(
    "⬇️ Download predictions (CSV)",
    data=out.to_csv(index=False),
    file_name="airbnb_price_predictions.csv",
    mime="text/csv",
)
