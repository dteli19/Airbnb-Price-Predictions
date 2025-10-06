# app.py — Airbnb Price Prediction (Structured & Beautified Workflow)
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
st.title("🏠 Airbnb Price Prediction — End-to-End Data Modeling Workflow")

# ----------------------------
# Helper Functions
# ----------------------------
DATA_PATH = Path("listings.csv")  # Hardcoded dataset path
TARGET = "price"

def clean_price_series(s: pd.Series) -> pd.Series:
    """Clean the price column (remove symbols, commas, etc.)"""
    if s.dtype == object:
        s = s.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def split_num_cat(df: pd.DataFrame, exclude=None):
    """Split columns into numeric and categorical types"""
    exclude = set(exclude or [])
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in exclude]
    return num_cols, cat_cols

def eval_regression(y_true, y_pred):
    """Compute MAE, RMSE, and R²"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ====================================================
# Section 1 — Overview, Problem Statement, About the Data
# ====================================================
st.header("Section 1 — Overview, Problem Statement, About the Data")

st.markdown("""
### **Overview**
This project develops a **regression model** to predict **Airbnb listing prices** using the publicly available *Inside Airbnb* dataset.  
It simulates an end-to-end data analytics workflow: **Data Loading → Preprocessing → EDA → Modeling → Evaluation.**

### **Problem Statement**
The objective is to predict the nightly **listing price** based on attributes like room type, capacity, location, and reviews —  
and determine which features most strongly influence price.

### **About the Data**
The dataset (`listings.csv`) includes:
- **Price-related attributes:** `price`, `minimum_nights`, `availability_365`
- **Property characteristics:** `accommodates`, `bedrooms`, `bathrooms`, `room_type`, `property_type`
- **Location details:** `latitude`, `longitude`, `neighbourhood`
- **Engagement indicators:** `number_of_reviews`, `review_scores_rating`

Each row represents one Airbnb listing for a given city.
""")

# ====================================================
# Section 2 — Actions / Workflow Steps
# ====================================================
st.header("Section 2 — Actions / Workflow Steps")

st.markdown("""
This project follows a structured **5-step modeling pipeline**:

1️⃣ **Data Loading** – Import and inspect the dataset (`listings.csv`).  
2️⃣ **Data Selection** – Identify relevant predictive variables and the target (`price`).  
3️⃣ **Data Preprocessing** – Handle missing values, convert data types, and prepare for modeling.  
4️⃣ **Exploratory Data Analysis (EDA)** – Explore target distributions and relationships between variables.  
5️⃣ **Prepare for Modeling & Evaluation** – Normalize data, build a **Linear Regression model**, and evaluate using **MAE**, **RMSE**, and **R²**.

The process ensures clean, interpretable results while maintaining reproducibility across cities and exports.
""")

# ====================================================
# Section 3 — Workflow Implementation
# ====================================================
st.header("Section 3 — Data Modeling Workflow")

# Step 1: Data Loading
st.subheader("Step 1: Data Loading")
if not DATA_PATH.exists():
    st.error("❌ File not found: `listings.csv`. Please add the dataset and rerun.")
    st.stop()

df = pd.read_csv(DATA_PATH, low_memory=False)
st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns")
st.dataframe(df.head(), use_container_width=True)

# Step 2: Data Selection
st.subheader("Step 2: Data Selection")
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
    features = [c for c in df.columns if c != TARGET][:20]

st.markdown(f"**Features selected for modeling:** {', '.join(features)}")
work = df[features + [TARGET]].copy()

# Step 3: Data Preprocessing
st.subheader("Step 3: Data Preprocessing")

num_cols, cat_cols = split_num_cat(work.drop(columns=[TARGET]))

# Handle missing data
for c in num_cols:
    work[c] = pd.to_numeric(work[c], errors="coerce").fillna(work[c].mean())
for c in cat_cols:
    work[c] = work[c].astype(object).fillna("Unknown")

before, after = len(work), len(work.dropna())
work = work.dropna()
st.info(f"Cleaned dataset: {after:,} valid rows (removed {before - after:,} incomplete entries).")

# Step 4: Exploratory Data Analysis
st.subheader("Step 4: Exploratory Data Analysis (EDA)")

# Price distribution
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Distribution of Price**")
    fig, ax = plt.subplots()
    sns.histplot(work[TARGET], bins=40, ax=ax, color="#66b3ff")
    ax.set_xlabel("Price")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Correlation heatmap
with c2:
    st.markdown("**Correlation (Numeric Features)**")
    corr_cols = [c for c in num_cols if c in work.columns] + [TARGET]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(work[corr_cols].corr(numeric_only=True), cmap="YlGnBu", annot=False, ax=ax)
    st.pyplot(fig)

# Step 5: Prepare for Modeling (Normalize & Split)
st.subheader("Step 5: Prepare for Modeling (Normalize & Split Data)")

X = work[features].copy()
y = work[TARGET].astype(float)
num_feats, cat_feats = split_num_cat(X)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
])

pipe = Pipeline([
    ("prep", preprocess),
    ("model", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ====================================================
# Section 4 — Results
# ====================================================
st.header("Section 4 — Results and Insights")

m1, m2, m3 = st.columns(3)
m1.metric("Mean Absolute Error (MAE)", f"{mae:,.2f}")
m2.metric("Root Mean Squared Error (RMSE)", f"{rmse:,.2f}")
m3.metric("R² Score", f"{r2:.3f}")

# Feature importance
st.subheader("Top Feature Influences on Price")
try:
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(cat_feats) if cat_feats else []
    feature_names = np.array(num_feats + list(cat_names))
    coefs = pipe.named_steps["model"].coef_
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
    coef_df["Abs"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Coefficient", y="Feature", data=coef_df, ax=ax, palette="viridis")
    st.pyplot(fig)
except Exception:
    st.info("Could not display feature influence chart.")

# Predicted vs Actual
st.subheader("Predicted vs Actual Prices")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5, color="#3b82f6")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
st.pyplot(fig)

st.markdown("""
### **Interpretation**
- **MAE & RMSE** quantify average prediction errors.  
- **R²** indicates how well the model explains price variation (closer to 1 = better).  
- **Feature Coefficients** show which features have the strongest influence on pricing.  
- **Predicted vs Actual Plot** helps visually assess model accuracy — points close to the red line indicate stronger fit.
""")

st.success("✅ Airbnb price prediction model successfully executed with interpretable results and clean workflow visualization!")
# app.py — Airbnb Price Prediction (Structured & Beautified Workflow)
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
st.title("🏠 Airbnb Price Prediction — End-to-End Data Modeling Workflow")

# ----------------------------
# Helper Functions
# ----------------------------
DATA_PATH = Path("data/listings.csv")  # Hardcoded dataset path
TARGET = "price"

def clean_price_series(s: pd.Series) -> pd.Series:
    """Clean the price column (remove symbols, commas, etc.)"""
    if s.dtype == object:
        s = s.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def split_num_cat(df: pd.DataFrame, exclude=None):
    """Split columns into numeric and categorical types"""
    exclude = set(exclude or [])
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in exclude]
    return num_cols, cat_cols

def eval_regression(y_true, y_pred):
    """Compute MAE, RMSE, and R²"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ====================================================
# Section 1 — Overview, Problem Statement, About the Data
# ====================================================
st.header("Section 1 — Overview, Problem Statement, About the Data")

st.markdown("""
### **Overview**
This project develops a **regression model** to predict **Airbnb listing prices** using the publicly available *Inside Airbnb* dataset.  
It simulates an end-to-end data analytics workflow: **Data Loading → Preprocessing → EDA → Modeling → Evaluation.**

### **Problem Statement**
The objective is to predict the nightly **listing price** based on attributes like room type, capacity, location, and reviews —  
and determine which features most strongly influence price.

### **About the Data**
The dataset (`listings.csv`) includes:
- **Price-related attributes:** `price`, `minimum_nights`, `availability_365`
- **Property characteristics:** `accommodates`, `bedrooms`, `bathrooms`, `room_type`, `property_type`
- **Location details:** `latitude`, `longitude`, `neighbourhood`
- **Engagement indicators:** `number_of_reviews`, `review_scores_rating`

Each row represents one Airbnb listing for a given city.
""")

# ====================================================
# Section 2 — Actions / Workflow Steps
# ====================================================
st.header("Section 2 — Actions / Workflow Steps")

st.markdown("""
This project follows a structured **5-step modeling pipeline**:

1️⃣ **Data Loading** – Import and inspect the dataset (`listings.csv`).  
2️⃣ **Data Selection** – Identify relevant predictive variables and the target (`price`).  
3️⃣ **Data Preprocessing** – Handle missing values, convert data types, and prepare for modeling.  
4️⃣ **Exploratory Data Analysis (EDA)** – Explore target distributions and relationships between variables.  
5️⃣ **Prepare for Modeling & Evaluation** – Normalize data, build a **Linear Regression model**, and evaluate using **MAE**, **RMSE**, and **R²**.

The process ensures clean, interpretable results while maintaining reproducibility across cities and exports.
""")

# ====================================================
# Section 3 — Workflow Implementation
# ====================================================
st.header("Section 3 — Data Modeling Workflow")

# Step 1: Data Loading
st.subheader("Step 1: Data Loading")
if not DATA_PATH.exists():
    st.error("❌ File not found: `data/listings.csv`. Please add the dataset and rerun.")
    st.stop()

df = pd.read_csv(DATA_PATH, low_memory=False)
st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns")
st.dataframe(df.head(), use_container_width=True)

# Step 2: Data Selection
st.subheader("Step 2: Data Selection")
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
    features = [c for c in df.columns if c != TARGET][:20]

st.markdown(f"**Features selected for modeling:** {', '.join(features)}")
work = df[features + [TARGET]].copy()

# Step 3: Data Preprocessing
st.subheader("Step 3: Data Preprocessing")

num_cols, cat_cols = split_num_cat(work.drop(columns=[TARGET]))

# Handle missing data
for c in num_cols:
    work[c] = pd.to_numeric(work[c], errors="coerce").fillna(work[c].mean())
for c in cat_cols:
    work[c] = work[c].astype(object).fillna("Unknown")

before, after = len(work), len(work.dropna())
work = work.dropna()
st.info(f"Cleaned dataset: {after:,} valid rows (removed {before - after:,} incomplete entries).")

# Step 4: Exploratory Data Analysis
st.subheader("Step 4: Exploratory Data Analysis (EDA)")

# Price distribution
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Distribution of Price**")
    fig, ax = plt.subplots()
    sns.histplot(work[TARGET], bins=40, ax=ax, color="#66b3ff")
    ax.set_xlabel("Price")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Correlation heatmap
with c2:
    st.markdown("**Correlation (Numeric Features)**")
    corr_cols = [c for c in num_cols if c in work.columns] + [TARGET]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(work[corr_cols].corr(numeric_only=True), cmap="YlGnBu", annot=False, ax=ax)
    st.pyplot(fig)

# Step 5: Prepare for Modeling (Normalize & Split)
st.subheader("Step 5: Prepare for Modeling (Normalize & Split Data)")

X = work[features].copy()
y = work[TARGET].astype(float)
num_feats, cat_feats = split_num_cat(X)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
])

pipe = Pipeline([
    ("prep", preprocess),
    ("model", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ====================================================
# Section 4 — Results
# ====================================================
st.header("Section 4 — Results and Insights")

m1, m2, m3 = st.columns(3)
m1.metric("Mean Absolute Error (MAE)", f"{mae:,.2f}")
m2.metric("Root Mean Squared Error (RMSE)", f"{rmse:,.2f}")
m3.metric("R² Score", f"{r2:.3f}")

# Feature importance
st.subheader("Top Feature Influences on Price")
try:
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(cat_feats) if cat_feats else []
    feature_names = np.array(num_feats + list(cat_names))
    coefs = pipe.named_steps["model"].coef_
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
    coef_df["Abs"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Coefficient", y="Feature", data=coef_df, ax=ax, palette="viridis")
    st.pyplot(fig)
except Exception:
    st.info("Could not display feature influence chart.")

# Predicted vs Actual
st.subheader("Predicted vs Actual Prices")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5, color="#3b82f6")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
st.pyplot(fig)

st.markdown("""
### **Interpretation**
- **MAE & RMSE** quantify average prediction errors.  
- **R²** indicates how well the model explains price variation (closer to 1 = better).  
- **Feature Coefficients** show which features have the strongest influence on pricing.  
- **Predicted vs Actual Plot** helps visually assess model accuracy — points close to the red line indicate stronger fit.
""")

st.success("✅ Airbnb price prediction model successfully executed with interpretable results and clean workflow visualization!")
