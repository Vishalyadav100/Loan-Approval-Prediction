# ==========================================================
# Loan Approval Prediction Dashboard
# download.py
# ==========================================================

import os
import streamlit as st
import pandas as pd


# ==========================================================
# Download Model Comparison Results
# ==========================================================

def download_results(results_df):

    csv = results_df.to_csv(index=False)

    st.download_button(

        label="📄 Download Model Comparison",

        data=csv,

        file_name="model_comparison.csv",

        mime="text/csv",

        key="download_results_csv"

    )


# ==========================================================
# Download Processed Dataset
# ==========================================================

def download_processed_dataset(processed_df):

    csv = processed_df.to_csv(index=False)

    st.download_button(

        label="📊 Download Processed Dataset",

        data=csv,

        file_name="processed_dataset.csv",

        mime="text/csv",

        key="processed_dataset_csv"

    )


# ==========================================================
# Download Trained Model
# ==========================================================

def download_model():

    model_path = "saved_models/Best_Model.pkl"

    if os.path.exists(model_path):

        with open(model_path, "rb") as file:

            st.download_button(

                label="🤖 Download Best Model",

                data=file,

                file_name="Best_Model.pkl",

                mime="application/octet-stream",

                key="download_best_model"

            )

    else:

        st.warning(

            "Train the model first."

        )


# ==========================================================
# Download Prediction Report
# ==========================================================

def download_prediction_report(input_df, prediction):

    report = input_df.copy()

    if prediction[0] == 1:

        report["Prediction"] = "Loan Approved"

    else:

        report["Prediction"] = "Loan Rejected"

    csv = report.to_csv(index=False)

    st.download_button(

        label="📋 Download Prediction Report",

        data=csv,

        file_name="prediction_report.csv",

        mime="text/csv",

        key="prediction_report"

    )


# ==========================================================
# Download Project Information
# ==========================================================

def download_project_info():

    info = pd.DataFrame({

        "Project":[
            "Loan Approval Prediction Dashboard"
        ],

        "Language":[
            "Python"
        ],

        "Framework":[
            "Streamlit"
        ],

        "Models":[
            "Logistic Regression, Decision Tree, Random Forest, XGBoost"
        ]

    })

    csv = info.to_csv(index=False)

    st.download_button(

        label="📑 Download Project Information",

        data=csv,

        file_name="project_information.csv",

        mime="text/csv",

        key="project_information"

    )

