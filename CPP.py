import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Load Dataset  
df = pd.read_excel("FAOSTAT_data.xlsx")

# DATA CLEANING & FEATURE ENGINEERING
df.drop(columns=["Domain Code", "Domain", "Area Code (M49)", "Element Code", "Item Code (CPC)", 
                 "Flag", "Flag Description", "Note"], inplace=True)

df = df[df["Element"].isin(["Area harvested", "Yield", "Production"])]
df = df.pivot_table(index=["Area", "Item", "Year"], columns="Element", values="Value").reset_index()
df.columns = ["Area", "Crop", "Year", "Area_Harvested", "Yield", "Production"]

# Handling Missing Values
df.dropna(inplace=True)

# Handling Outliers using IQR
for col in ["Area_Harvested", "Yield", "Production"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# FEATURE ENGINEERING
df["Area_Yield_Interaction"] = df["Area_Harvested"] * df["Yield"]

# Streamlit App
st.title("🌾 Crop Production Prediction")

# Exploratory Data Analysis (EDA) Section
st.sidebar.header("📊 Exploratory Data Analysis")
eda_option = st.sidebar.selectbox("Choose an analysis", ["None", "Production Distribution", "Scatter Plots", "Feature Correlation"])

if eda_option == "Production Distribution":
    st.subheader("📌 Production Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Production"], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

elif eda_option == "Scatter Plots":
    st.subheader("📌 Area Harvested vs Production")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df["Area_Harvested"], y=df["Production"], ax=ax)
    st.pyplot(fig)

    st.subheader("📌 Yield vs Production")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df["Yield"], y=df["Production"], ax=ax)
    st.pyplot(fig)

elif eda_option == "Feature Correlation":
    st.subheader("📌 Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df[["Area_Harvested", "Yield", "Production"]].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# DATA SPLITTING
X = df[["Area_Harvested", "Yield", "Area_Yield_Interaction"]]
y = df["Production"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODEL TRAINING
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
}

# Train and Store Performance
model_performance = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    model_performance[name] = r2

# Save Best Model
best_model = max(model_performance, key=model_performance.get)
best_model_instance = models[best_model]

with open("crop_production_model.pkl", "wb") as f:
    pickle.dump(best_model_instance, f)

# Prediction Section
st.sidebar.header("🌱 Make a Prediction")
area_harvested = st.sidebar.number_input("Enter Area Harvested (ha)", min_value=0.0, step=1.0)
yield_value = st.sidebar.number_input("Enter Yield (kg/ha)", min_value=0.0, step=1.0)
model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest", "XGBoost"])

if st.sidebar.button("Predict"):
    with open("crop_production_model.pkl", "rb") as f:
        model = pickle.load(f)

    input_data = scaler.transform([[area_harvested, yield_value, area_harvested * yield_value]])
    prediction = model.predict(input_data)
    st.subheader(f"🌾 Predicted Crop Production: *{prediction[0]:,.2f} tons*")

# Display Model Performance
st.subheader("📈 Model Performance")
for name, r2 in model_performance.items():
    st.write(f"*{name}:* R² Score = {r2:.4f}")

# Run Streamlit
if "__name_" == "_main_":
    st.write("✅ Streamlit App Running")
