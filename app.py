import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Bhutan Health Indicators – Tuberculosis, Nutrition & NCDs")

st.write(
    "This demo app uses **WHO health indicators for Bhutan** "
    "to explore data and build a simple prediction model for the indicator value."
)

# -------------------------------------------------------------------
# File paths – rename these if your filenames are different
# -------------------------------------------------------------------
TB_FILE = "tb_indicators_btn.csv"
NUTRITION_FILE = "nutrition_indicators_btn.csv"
NCD_FILE = "ncd_indicators_btn.csv"

# -------------------------------------------------------------------
# 1. Data loading
# -------------------------------------------------------------------
@st.cache_data
def load_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name == "Tuberculosis":
        path = TB_FILE
    elif dataset_name == "Nutrition":
        path = NUTRITION_FILE
    else:
        path = NCD_FILE

    df = pd.read_csv(path)

    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # MOST WHO humdata CSVs have a numeric column called 'value'
    # If your file uses something else (e.g. 'Numeric'), change it here.
    if "value" not in df.columns:
        st.warning(
            "Column 'value' not found. Please check your CSV and adjust the code "
            "to use the correct target column name."
        )

    # Try convert year-like columns to numeric (if present)
    for col in ["year", "time", "time_period"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure value is numeric
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Drop rows where target is missing
    if "value" in df.columns:
        df = df.dropna(subset=["value"])

    return df

# -------------------------------------------------------------------
# 2. Model training (simple regressor on year + a few basic cols)
# -------------------------------------------------------------------
@st.cache_resource
def train_model(df: pd.DataFrame):
    df = df.copy()

    # Build features: use year if available + encode simple categoricals
    drop_cols = []

    # If a clear id column exists, drop it from features
    for c in ["country", "iso3", "indicator_code", "indicator", "indicator_name"]:
        if c in df.columns:
            drop_cols.append(c)

    if "value" not in df.columns:
        return None, None

    y = df["value"]
    X = df.drop(columns=drop_cols + ["value"], errors="ignore")

    # One-hot encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    if X.empty:
        return None, None

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    return model, X.columns.tolist()

# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
dataset_choice = st.sidebar.selectbox(
    "Select dataset",
    ["Tuberculosis", "Nutrition", "Noncommunicable diseases"],
)

menu = st.sidebar.selectbox(
    "Navigate",
    ["Dataset Overview", "Visualizations", "Train Model Summary", "Predict Indicator Value"],
)

data = load_dataset(dataset_choice)
model, feature_columns = train_model(data)

# Only numeric columns
numeric_data = data.select_dtypes(include=["number"])

# -------------------------------------------------------------------
# Dataset Overview
# -------------------------------------------------------------------
if menu == "Dataset Overview":
    st.header(f"{dataset_choice} Indicators – Dataset Overview (Bhutan)")

    st.subheader("Preview")
    st.dataframe(data.head())

    st.subheader("Summary Statistics")
    st.dataframe(data.describe(include="all"))

    if "value" in data.columns:
        st.subheader("Distribution of 'value'")
        fig, ax = plt.subplots()
        ax.hist(data["value"], bins=20)
        ax.set_xlabel("value")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

# -------------------------------------------------------------------
# Visualizations
# -------------------------------------------------------------------
elif menu == "Visualizations":
    st.header(f"{dataset_choice} – Visualizations")

    viz_type = st.selectbox(
        "Choose chart type:",
        [
            "Correlation Heatmap",
            "Line Chart (Year vs Value)",
            "Bar Chart (Categorical counts)",
            "Histogram (numeric column)",
            "Scatter Plot (numeric vs numeric)",
        ]
    )

    if viz_type == "Correlation Heatmap":
        if numeric_data.shape[1] < 2:
            st.warning("Not enough numeric columns for a correlation heatmap.")
        else:
            st.subheader("Correlation Heatmap (numeric columns)")
            corr = numeric_data.corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    elif viz_type == "Line Chart (Year vs Value)":
        if "year" in data.columns and "value" in data.columns:
            st.subheader("Year vs Value")
            line_df = data[["year", "value"]].dropna().sort_values("year")
            line_df = line_df.set_index("year")
            st.line_chart(line_df)
        else:
            st.warning("Columns 'year' and/or 'value' not found in this dataset.")

    elif viz_type == "Bar Chart (Categorical counts)":
        cat_cols = data.select_dtypes(include=["object"]).columns.tolist()
        if not cat_cols:
            st.warning("No categorical columns found.")
        else:
            col = st.selectbox("Select categorical column:", cat_cols)
            st.bar_chart(data[col].value_counts())

    elif viz_type == "Histogram (numeric column)":
        if numeric_data.empty:
            st.warning("No numeric columns found.")
        else:
            col = st.selectbox("Select numeric column:", numeric_data.columns)
            fig, ax = plt.subplots()
            ax.hist(data[col], bins=20)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    elif viz_type == "Scatter Plot (numeric vs numeric)":
        if numeric_data.shape[1] < 2:
            st.warning("Need at least two numeric columns.")
        else:
            x_col = st.selectbox("X-axis:", numeric_data.columns)
            y_col = st.selectbox("Y-axis:", numeric_data.columns, index=1 if len(numeric_data.columns) > 1 else 0)
            fig, ax = plt.subplots()
            ax.scatter(data[x_col], data[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig)

# -------------------------------------------------------------------
# Train Model Summary
# -------------------------------------------------------------------
elif menu == "Train Model Summary":
    st.header(f"{dataset_choice} – Model Training Summary")

    if model is None or not feature_columns:
        st.warning(
            "Could not train model. Check that 'value' exists and there are usable features."
        )
    else:
        st.subheader("Feature Importances (Random Forest Regressor)")

        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))
        st.dataframe(importance_df)

# -------------------------------------------------------------------
# Predict Indicator Value
# -------------------------------------------------------------------
elif menu == "Predict Indicator Value":
    st.header(f"{dataset_choice} – Predict Indicator Value (Demo)")

    if model is None or not feature_columns:
        st.warning(
            "Model is not available. Please ensure data has a 'value' column "
            "and at least one feature."
        )
    else:
        st.write(
            "This section lets you input numeric features and get a predicted value. "
            "For categorical one-hot features, default values are used."
        )

        # Use numeric columns only for sliders
        numeric_cols = numeric_data.columns.tolist()
        # Don't include target
        numeric_cols = [c for c in numeric_cols if c != "value"]

        user_input = {}
        for col in numeric_cols:
            col_min = float(data[col].min())
            col_max = float(data[col].max())
            default = float(data[col].median())
            user_input[col] = st.slider(
                col,
                min_value=col_min,
                max_value=col_max,
                value=default,
            )

        if st.button("Predict"):
            input_df = pd.DataFrame([user_input])

            # Match same preprocessing as training (one-hot & column order)
            input_df = pd.get_dummies(input_df, drop_first=True)
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)

            pred = model.predict(input_df)[0]
            st.success(f"Predicted indicator value: {pred:.2f}")
