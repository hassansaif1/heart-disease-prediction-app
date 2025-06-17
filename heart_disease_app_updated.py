import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


st.set_page_config(page_title="Heart Disease App", layout="wide")
# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Prediction", "Classification", "About"])

# Dataset upload
st.sidebar.markdown("### Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load and cache dataset
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df.dropna()
    return df

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if "TenYearCHD" not in df.columns:
        st.error("Your dataset must include the column 'TenYearCHD' as the target.")
        st.stop()
    X = df.drop("TenYearCHD", axis=1)
    y = df["TenYearCHD"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    st.warning("👈 Please upload your dataset to get started.")
    st.stop()
# -------------------- Page 1: Home --------------------
if page == "Home":
    st.title("💖 Web Application for Heart Disease using Machine Learning")

    st.markdown("""
    ### 🩺 What is Heart Disease?
    Heart disease describes a range of conditions that affect your heart, including blood vessel disease, arrhythmias, and congenital defects.

    ### ⚠️ Common Symptoms:
    - Chest pain or discomfort  
    - Shortness of breath  
    - Fatigue  
    - Irregular heartbeat  
    - Swelling in legs or abdomen

    ### 📊 Age-wise Distribution (Heart Disease Rate %):
    """)


    # Define age bins with proper labels
    bins = [20, 30, 40, 50, 60, 70, 80]
    labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
    age_bins = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    age_group_stats = df.groupby(age_bins)["TenYearCHD"].mean().reset_index()
    age_group_stats["Heart Disease Rate (%)"] = (age_group_stats["TenYearCHD"] * 100).round(2)
    st.dataframe(age_group_stats.rename(columns={"age": "Age Group"}).drop(columns="TenYearCHD"))

    st.markdown("### 🛡️ Prevention Tips:")
    st.markdown("""
    - Eat a healthy diet
    - Exercise regularly
    - Quit smoking
    - Manage stress
    - Regular health screenings
    """)

# -------------------- Page 2: EDA --------------------
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

    st.subheader("📊 Bar Plot: Heart Disease by Gender")
    if 'male' in df.columns:
        fig, ax = plt.subplots()
        sns.barplot(x='male', y='TenYearCHD', data=df, ax=ax)
        st.pyplot(fig)

    st.subheader("📉 Line Chart: Age vs Heart Disease Rate")
    age_line = df.groupby("age")["TenYearCHD"].mean()
    st.line_chart(age_line)
# -------------------- Page 3: Prediction --------------------
elif page == "Prediction":
    st.title("🤖 Predict Heart Disease")

    st.sidebar.header("🔍 Select Model")
    model_name = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "Decision Tree", "SVM", "Gradient Boosting"])

    # Initialize model and accuracy outside the if/elif blocks
    # This ensures they are always defined
    model = None
    acc = 0.0

    if model_name == "Logistic Regression":
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(X_train)
        x_test_scaled = scaler.transform(X_test)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        acc = accuracy_score(y_test, y_pred)

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)  # Added max_depth for better control
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

    elif model_name == "SVM":
        model = SVC(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

    else: # Gradient Boosting
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

    st.markdown("### 🧾 Enter Patient Info")

    # Inputs with consistent numeric types (all float or all int)
    # For binary variables: integer input 0 or 1
    user_input = {}

    user_input['male'] = st.number_input(
        "Male (0 = Female, 1 = Male)", min_value=0, max_value=1, step=1, value=int(round(df['male'].mean()))
    )
    user_input['age'] = st.number_input(
        "Age", min_value=20, max_value=120, value=int(round(df['age'].mean()))
    )
    user_input['cigsPerDay'] = st.number_input(
        "Cigarettes Per Day", min_value=0, max_value=100, value=int(round(df['cigsPerDay'].mean()))
    )
    user_input['BPMeds'] = st.number_input(
        "On Blood Pressure Medication (0 = No, 1 = Yes)", min_value=0, max_value=1, step=1, value=int(round(df['BPMeds'].mean()))
    )
    user_input['prevalentStroke'] = st.number_input(
        "Prevalent Stroke (0 = No, 1 = Yes)", min_value=0, max_value=1, step=1, value=int(round(df['prevalentStroke'].mean()))
    )
    user_input['prevalentHyp'] = st.number_input(
        "Prevalent Hypertension (0 = No, 1 = Yes)", min_value=0, max_value=1, step=1, value=int(round(df['prevalentHyp'].mean()))
    )
    user_input['totChol'] = st.number_input(
        "Total Cholesterol (mg/dL)", min_value=100.0, max_value=500.0, value=float(round(df['totChol'].mean(), 2))
    )
    user_input['sysBP'] = st.number_input(
        "Systolic Blood Pressure (mmHg)", min_value=90.0, max_value=250.0, value=float(round(df['sysBP'].mean(), 2))
    )
    user_input['diaBP'] = st.number_input(
        "Diastolic Blood Pressure (mmHg)", min_value=60.0, max_value=150.0, value=float(round(df['diaBP'].mean(), 2))
    )
    user_input['BMI'] = st.number_input(
        "Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=float(round(df['BMI'].mean(), 2))
    )

    if st.button("🚨 Predict"):
        try:
            input_data = {
                'male': int(user_input['male']),
                'age': int(user_input['age']),
                'cigsPerDay': int(user_input['cigsPerDay']),
                'BPMeds': int(user_input['BPMeds']),
                'prevalentStroke': int(user_input['prevalentStroke']),
                'prevalentHyp': int(user_input['prevalentHyp']),
                'totChol': float(user_input['totChol']),
                'sysBP': float(user_input['sysBP']),
                'diaBP': float(user_input['diaBP']),
                'BMI': float(user_input['BMI']),
            }

            input_df = pd.DataFrame([input_data])

            # Fill missing columns with training data means if any columns are missing
            for col in X_train.columns:
                if col not in input_df.columns:
                    input_df[col] = X_train[col].mean()
            # Reorder columns to match training data
            input_df = input_df[X_train.columns]

            # Scale input data if Logistic Regression is selected
            if model_name == "Logistic Regression":
                input_df_scaled = scaler.transform(input_df)
                pred = model.predict(input_df_scaled)[0]
            else:
                pred = model.predict(input_df)[0]

            st.subheader("🔍 Result:")
            if pred == 1:
                st.error("⚠️ HIGH risk of heart disease.")
            else:
                st.success("✅ LOW risk of heart disease.")
            st.info(f"Model Accuracy: {acc * 100:.2f}%")

        except ValueError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            
# -------------------- Page 4: Classification --------------------
elif page == "Classification":
    st.title("📋 Model Evaluation")

    st.sidebar.header("🔎 Evaluation Type")
    eval_type = st.sidebar.radio("Choose", ["Training Data", "Testing Data"])

    model_name = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "SVM", "Gradient Boosting"])

    # Initialize scaler outside if block, only needed for Logistic Regression
    scaler = None
    if model_name == "Logistic Regression":
        # Create and fit scaler for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression(max_iter=1000, random_state=42) # Added random_state
        model.fit(X_train_scaled, y_train) # Fit on scaled data
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42) # Added random_state
        model.fit(X_train, y_train)
    elif model_name == "SVM":
        model = SVC(random_state=42) # Added random_state
        model.fit(X_train, y_train)
    else: # Gradient Boosting
        model = GradientBoostingClassifier(random_state=42) # Added random_state
        model.fit(X_train, y_train)

    # Determine which data to use for evaluation and prediction
    if eval_type == "Training Data":
        y_eval = y_train
        if model_name == "Logistic Regression":
            y_pred = model.predict(X_train_scaled) # Predict on scaled training data
        else:
            y_pred = model.predict(X_train)
    else: # Testing Data
        y_eval = y_test
        if model_name == "Logistic Regression":
            y_pred = model.predict(X_test_scaled) # Predict on scaled testing data
        else:
            y_pred = model.predict(X_test)

    st.subheader("📊 Classification Report")
    report = classification_report(y_eval, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'))

    st.subheader("🔁 Confusion Matrix")
    cm = confusion_matrix(y_eval, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)            
