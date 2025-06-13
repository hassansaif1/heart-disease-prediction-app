
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Heart Disease App", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Prediction", "Classification", "About"])

# Dataset upload
st.sidebar.markdown("### Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.dropna(inplace=True)
    return df

def preprocess(df):
    df['education'] = df['education'].astype('Int64')
    df['cigsPerDay'] = df['cigsPerDay'].astype('Int64')
    df['BPMeds'] = df['BPMeds'].astype('Int64')
    return df

def encode_age(age):
    if age <= 40:
        return 0
    elif age <= 55:
        return 1
    else:
        return 2

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if "TenYearCHD" not in df.columns:
        st.error("Your dataset must include the column 'TenYearCHD' as the target.")
        st.stop()
    df = preprocess(df)
    df["age_group"] = df["age"].apply(encode_age)

    X = df.drop(["TenYearCHD"], axis=1)
    y = df["TenYearCHD"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

else:
    st.warning("👈 Please upload your dataset to get started.")
    st.stop()

# -------------------- Pages --------------------
if page == "Home":
    st.title("💖 Web Application for Heart Disease using Machine Learning")
    st.markdown("""
    ### 🩺 What is Heart Disease?
    Heart disease includes blood vessel issues, arrhythmias, and congenital defects.

    ### ⚠️ Common Symptoms:
    - Chest pain
    - Shortness of breath
    - Fatigue
    - Irregular heartbeat
    - Swelling

    ### 📊 Age-wise Distribution:
    """)
    bins = [20, 30, 40, 50, 60, 70, 80]
    labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
    age_bins = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    age_stats = df.groupby(age_bins)["TenYearCHD"].mean().reset_index()
    age_stats["Heart Disease Rate (%)"] = (age_stats["TenYearCHD"] * 100).round(2)
    st.dataframe(age_stats.drop(columns="TenYearCHD"))

elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.subheader("📈 Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif page == "Prediction":
    st.title("🤖 Predict Heart Disease")
    st.markdown("### 🧾 Enter Patient Info")

    user_input = {}
    for col in X.columns:
        dtype = df[col].dtype
        if dtype == "int64" or dtype.name.startswith("Int"):
            user_input[col] = st.number_input(col, min_value=0, value=int(df[col].mean()))
        else:
            user_input[col] = st.number_input(col, value=float(df[col].mean()))

    if st.button("🚨 Predict"):
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.subheader("🔍 Result:")
        if prediction == 1:
            st.error("⚠️ HIGH risk of heart disease.")
        else:
            st.success("✅ LOW risk of heart disease.")
        st.info(f"Model Accuracy: {accuracy * 100:.2f}%")

elif page == "Classification":
    st.title("📋 Model Evaluation")
    st.subheader("📊 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("🔁 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

elif page == "About":
    st.title("📘 About This Project")
    st.markdown("""
    ### 💡 Overview
    This app visualizes, explores, and predicts heart disease risk using logistic regression.

    ### 🧰 Technologies
    - Python, Pandas, NumPy
    - Scikit-learn
    - Streamlit
    - Seaborn, Matplotlib

    ⚠️ Educational use only, not for medical diagnosis.
    """)
