import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

# ========== PAGE CONFIGURATION ==========
st.set_page_config(page_title="Crop Production Prediction", page_icon="🌾", layout="wide")

# Apply Background Color
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5; /* Light Gray */
        }
    </style>
""", unsafe_allow_html=True)

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    df = pd.read_excel("FAOSTAT_data.xlsx")
    df.drop(columns=["Domain Code", "Domain", "Area Code (M49)", "Element Code", "Item Code (CPC)", 
                     "Flag", "Flag Description", "Note"], inplace=True)
    
    df = df[df["Element"].isin(["Area harvested", "Yield", "Production"])]
    df = df.pivot_table(index=["Area", "Item", "Year"], columns="Element", values="Value").reset_index()
    df.columns = ["Area", "Crop", "Year", "Area_Harvested", "Yield", "Production"]
    df.dropna(inplace=True)

    # Handling Outliers using IQR
    for col in ["Area_Harvested", "Yield", "Production"]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    df["Log_Production"] = np.log1p(df["Production"])
    df["Area_Yield_Interaction"] = df["Area_Harvested"] * df["Yield"]
    return df

df = load_data()

# ========== TRAIN MODEL (Optimized) ==========
@st.cache_resource
def train_model():
    model_file = "trained_models.pkl"

    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            trained_models, model_performance, scaler = pickle.load(f)
    else:
        X = df[["Area_Harvested", "Yield", "Area_Yield_Interaction"]]
        y = df["Production"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        }

        model_performance, trained_models = {}, {}

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            model_performance[name] = r2_score(y_test, y_pred)
            trained_models[name] = model

        with open(model_file, "wb") as f:
            pickle.dump((trained_models, model_performance, scaler), f)

    return trained_models, model_performance, scaler

models, model_performance, scaler = train_model()

# ========== EDA VISUALIZATION ==========
@st.cache_data
def generate_eda():
    plots = {}

    # Optimal figure size and DPI for better visibility
    fig_width, fig_height = 6, 4
    dpi_value = 120

    # 1. Production Distribution
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_value)
    sns.histplot(df["Production"], bins=40, kde=True, ax=ax)
    ax.set_title("Production Distribution", fontsize=12)
    ax.tick_params(axis='x', labelsize=10, rotation=30)
    plt.tight_layout()
    plots["Production Distribution"] = fig

    # 2. Area Harvested vs Production
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_value)
    sns.scatterplot(x=df["Area_Harvested"], y=df["Production"], alpha=0.5, ax=ax)  # Reduce opacity for clarity
    ax.set_title("Area Harvested vs Production", fontsize=12)
    plt.tight_layout()
    plots["Area Harvested vs Production"] = fig

    # 3. Yield vs Production
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_value)
    sns.scatterplot(x=df["Yield"], y=df["Production"], alpha=0.5, ax=ax)
    ax.set_title("Yield vs Production", fontsize=12)
    plt.tight_layout()
    plots["Yield vs Production"] = fig

    # 4. Feature Correlation Heatmap
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_value)
    sns.heatmap(df[["Area_Harvested", "Yield", "Production"]].corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=12)
    plt.tight_layout()
    plots["Feature Correlation Heatmap"] = fig

    # 5. Boxplot for Outlier Detection
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_value)
    sns.boxplot(data=df[["Area_Harvested", "Yield", "Production"]], ax=ax)
    ax.set_title("Outlier Analysis (Boxplot)", fontsize=12)
    plt.tight_layout()
    plots["Outlier Analysis"] = fig

    # 6. Log Production Distribution
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_value)
    sns.histplot(df["Log_Production"], bins=40, kde=True, ax=ax)
    ax.set_title("Log Production Distribution", fontsize=12)
    plt.tight_layout()
    plots["Log Production Distribution"] = fig

    # 7. Yearly Production Trend
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_value)
    df.groupby("Year")["Production"].sum().plot(ax=ax, marker="o", linestyle="-")
    ax.set_title("Yearly Production Trend", fontsize=12)
    plt.tight_layout()
    plots["Yearly Production Trend"] = fig

    # 8. Crop-wise Production
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_value)
    sns.boxplot(x="Crop", y="Production", data=df, ax=ax)
    ax.set_title("Crop-wise Production Distribution", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plots["Crop-wise Production"] = fig

    # 9. Area-wise Production
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_value)
    df.groupby("Area")["Production"].sum().nlargest(10).plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Top 10 Areas by Production", fontsize=12)
    plt.xticks(rotation=45, ha="right")  # Prevents label overlap
    plt.tight_layout()
    plots["Top 10 Areas by Production"] = fig

    # 10. Scatter Plot of Area_Harvested vs Yield
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_value)
    sns.scatterplot(x=df["Area_Harvested"], y=df["Yield"], alpha=0.5, ax=ax)
    ax.set_title("Area Harvested vs Yield", fontsize=12)
    plt.tight_layout()
    plots["Area Harvested vs Yield"] = fig

    return plots

eda_plots = generate_eda()

# ========== EDA DISPLAY ==========
st.header("📊 Exploratory Data Analysis")

# Use columns to arrange visualizations in a better layout
col1, col2 = st.columns(2)

# Display each plot in alternating columns
for i, (plot_name, plot_fig) in enumerate(eda_plots.items()):
    if i % 2 == 0:
        with col1:
            st.subheader(plot_name)
            st.pyplot(plot_fig)
    else:
        with col2:
            st.subheader(plot_name)
            st.pyplot(plot_fig)



# ========== LAYOUT ==========
st.markdown("<h1 style='text-align: center;'>🌾 Crop Production Prediction</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

# ========== MAIN SECTION (PREDICTION) ==========
with col1:
    st.markdown("### Enter Values to Predict Production")

    with st.form(key="prediction_form"):
        area_harvested = st.number_input("Enter Area Harvested (ha)", min_value=0.0, step=0.1, value=10.0)
        yield_value = st.number_input("Enter Yield (kg/ha)", min_value=0.0, step=0.1, value=5.0)
        model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest", "XGBoost"])

        submit_button = st.form_submit_button("📈 Predict")

        if submit_button:
            input_data = scaler.transform([[area_harvested, yield_value, area_harvested * yield_value]])
            prediction = models[model_choice].predict(input_data)
            st.success(f"Predicted Crop Production: {prediction[0]:,.2f} tons")

# ========== MODEL PERFORMANCE ==========
with col2:
    st.header("📊 Model Performance")
    for name, r2 in model_performance.items():
        st.markdown(f"{name}: R² Score = {r2:.4f}")

# ========== EDA SECTION ==========
st.header("📊 Exploratory Data Analysis")
eda_option = st.selectbox("Choose an analysis:", list(eda_plots.keys()))
st.pyplot(eda_plots[eda_option])
