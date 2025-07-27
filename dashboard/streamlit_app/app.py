# dashboard/streamlit_app/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.inference import predict_failure, apply_risk_level
from src.preprocess import preprocess_data

# Configure the Streamlit page
st.set_page_config(page_title="🏭 Equipment Failure Dashboard", layout="wide")
st.title(" Factory Equipment Failure Prediction System")

#  Note for Recruiters
st.info(
    " **Note for Users:**\n"
    "To test the system, please first download the sample CSV file below. "
    "This file contains the correct sensor data format required by the model.\n\n"
    "** Please do not upload any other CSV — it may result in errors.**"
)

# ✅ Download sample CSV
try:
    sample_df = pd.read_csv("data/synthetic_sensor_data.csv")  # Path to sample file
    csv = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Sample Sensor CSV",
        data=csv,
        file_name="synthetic_sensor_data.csv",
        mime="text/csv"
    )
except FileNotFoundError:
    st.warning("⚠️ Sample CSV file not found. Please ensure it exists in the 'data/' directory.")

# 📤 Upload CSV
uploaded_file = st.file_uploader("📁 Upload sensor data (CSV)", type="csv")

if uploaded_file is not None:
    try:
        # Step 1: Read uploaded file
        input_df = pd.read_csv(uploaded_file)
        st.subheader("🗂️ Raw Uploaded Data")
        st.dataframe(input_df, use_container_width=True)

        # Step 2: Preprocess
        X_processed = preprocess_data(input_df)

        # Step 3: Predict probabilities
        probabilities = predict_failure(X_processed)

        # Step 4: Add predictions
        input_df['Failure_Probability'] = probabilities.round(3)
        input_df['Risk_Level'] = input_df['Failure_Probability'].apply(apply_risk_level)

        st.success("✅ Prediction successful!")

        # Step 5: Color-coded table
        def color_rows(row):
            if row['Risk_Level'] == 'High':
                return ['background-color: red; color: white'] * len(row)
            elif row['Risk_Level'] == 'Medium':
                return ['background-color: orange; color: black'] * len(row)
            elif row['Risk_Level'] == 'Low':
                return ['background-color: green; color: white'] * len(row)
            else:
                return [''] * len(row)

        st.subheader("📋 Prediction Results")
        st.dataframe(input_df.style.apply(color_rows, axis=1), use_container_width=True)

        # Step 6: Risk Level Bar Chart
        st.subheader("📊 Risk Level Distribution")
        risk_counts = input_df['Risk_Level'].value_counts().reindex(['Low', 'Medium', 'High']).fillna(0)
        fig1, ax1 = plt.subplots()
        bars = ax1.bar(risk_counts.index, risk_counts.values, color=['green', 'orange', 'red'])
        ax1.set_ylabel("Count")
        ax1.set_title("Risk Level Distribution")
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center')
        st.pyplot(fig1)

        # Step 7: Correlation Heatmap
        st.subheader("📌 Feature Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        corr = input_df.select_dtypes(include='number').corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Error: {e}")
