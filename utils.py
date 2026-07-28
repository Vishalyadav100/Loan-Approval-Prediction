# ==========================================================
# Loan Approval Prediction Dashboard
# utils.py
# ==========================================================

import streamlit as st

# ==========================================================
# Footer
# ==========================================================

def footer():

    st.divider()

    st.markdown(

        """
        <center>

        Developed by <b>Vishal Yadav</b>

        <br>

        Loan Approval Prediction Dashboard

        <br>

        Powered by Streamlit & Scikit-Learn

        </center>

        """,

        unsafe_allow_html=True

    )


# ==========================================================
# About Project
# ==========================================================

def about_project():

    st.header("ℹ About Project")

    st.write("""

This project predicts whether a loan application will be approved or rejected using Machine Learning.

### Features

- Dataset Analysis
- Data Preprocessing
- Multiple ML Models
- Model Comparison
- Loan Prediction
- Visualization
- Download Trained Model

### Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib

""")

# ==========================================================
# Project Information Card
# ==========================================================

def project_information():

    st.info("""

Project Name :
Loan Approval Prediction Dashboard

Dataset :
614 Rows × 13 Columns

Target Column :
Loan_Status

""")

# ==========================================================
# Success Message
# ==========================================================

def success_message(message):

    st.success(message)

# ==========================================================
# Error Message
# ==========================================================

def error_message(message):

    st.error(message)

# ==========================================================
# Warning Message
# ==========================================================

def warning_message(message):

    st.warning(message)