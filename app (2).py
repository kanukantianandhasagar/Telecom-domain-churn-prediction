import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("Customer Churn Prediction App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("churn_dataset.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.drop(columns=["customerID"], inplace=True)
    return df

df = load_data()

# Preprocessing
target = "Churn"
X = df.drop(columns=[target])
y = df[target].apply(lambda x: 1 if x == "Yes" else 0)

X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
@st.cache_resource
def train_model():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Input interface
st.sidebar.header("Input Customer Details")
user_input = {}
for col in X.columns:
    if "Yes" in col or "No" in col:
        user_input[col] = st.sidebar.selectbox(col, [0, 1])
    elif X[col].dtype in [np.float64, np.int64]:
        user_input[col] = st.sidebar.number_input(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    else:
        user_input[col] = st.sidebar.text_input(col)

# Convert to DataFrame
input_df_raw = pd.DataFrame([user_input])
input_df_encoded = pd.get_dummies(input_df_raw)
input_df_final = input_df_encoded.reindex(columns=X.columns, fill_value=0)

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_df_final)[0]
    st.subheader("Prediction Result:")
    st.write("Customer is likely to **churn**." if prediction else "Customer is likely to **stay**.")

# Show data
if st.checkbox("Show Raw Dataset"):
    st.write(df)
