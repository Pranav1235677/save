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

# ========== TRAIN MODEL ==========
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
    fig_size = (3, 1.5)  # Adjusted for better clarity
    dpi_value = 150  # High clarity

    # 1. Production Distribution
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi_value)
    sns.histplot(df["Production"], bins=40, kde=True, ax=ax)
    ax.set_title("Production Distribution", fontsize=10)
    ax.tick_params(axis='x', labelsize=8, rotation=30)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plots["Production Distribution"] = fig

    # 2. Area Harvested vs Production
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi_value)
    sns.scatterplot(x=df["Area_Harvested"], y=df["Production"], ax=ax)
    ax.set_title("Area Harvested vs Production", fontsize=10)
    plt.tight_layout()
    plots["Area Harvested vs Production"] = fig

    # 3. Yield vs Production
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi_value)
    sns.scatterplot(x=df["Yield"], y=df["Production"], ax=ax)
    ax.set_title("Yield vs Production", fontsize=10)
    plt.tight_layout()
    plots["Yield vs Production"] = fig

    # 4. Feature Correlation Heatmap
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi_value)
    sns.heatmap(df[["Area_Harvested", "Yield", "Production"]].corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=10)
    plt.tight_layout()
    plots["Feature Correlation Heatmap"] = fig

    # 5. Boxplot for Outlier Detection
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi_value)
    sns.boxplot(data=df[["Area_Harvested", "Yield", "Production"]], ax=ax)
    ax.set_title("Outlier Analysis (Boxplot)", fontsize=10)
    plt.tight_layout()
    plots["Outlier Analysis"] = fig

    # 6. Log Production Distribution
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi_value)
    sns.histplot(df["Log_Production"], bins=40, kde=True, ax=ax)
    ax.set_title("Log Production Distribution", fontsize=10)
    plt.tight_layout()
    plots["Log Production Distribution"] = fig

    # 7. Yearly Production Trend
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi_value)
    df.groupby("Year")["Production"].sum().plot(ax=ax)
    ax.set_title("Yearly Production Trend", fontsize=10)
    plt.tight_layout()
    plots["Yearly Production Trend"] = fig

    # 8. Crop-wise Production
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi_value)
    sns.boxplot(x="Crop", y="Production", data=df, ax=ax)
    ax.set_title("Crop-wise Production Distribution", fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plots["Crop-wise Production"] = fig

    # 9. Area-wise Production
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi_value)
    df.groupby("Area")["Production"].sum().nlargest(10).plot(kind="bar", ax=ax)
    ax.set_title("Top 10 Areas by Production", fontsize=10)
    plt.tight_layout()
    plots["Top 10 Areas by Production"] = fig

    # 10. Scatter Plot of Area_Harvested vs Yield
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi_value)
    sns.scatterplot(x=df["Area_Harvested"], y=df["Yield"], ax=ax)
    ax.set_title("Area Harvested vs Yield", fontsize=10)
    plt.tight_layout()
    plots["Area Harvested vs Yield"] = fig

    return plots

eda_plots = generate_eda()

# ========== EDA DISPLAY ==========
st.header("📊 Exploratory Data Analysis")
eda_option = st.selectbox("Choose an analysis:", list(eda_plots.keys()))
st.pyplot(eda_plots[eda_option])
