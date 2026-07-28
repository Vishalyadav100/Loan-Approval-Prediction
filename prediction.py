# ==========================================================
# Loan Approval Prediction Dashboard
# prediction.py (Part A)
# ==========================================================

import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# Customer Input
# ==========================================================

def customer_input():

    st.subheader("Applicant Details")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=1500
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1,
        value=120
    )

    loan_term = st.selectbox(
        "Loan Amount Term",
        [12,36,60,84,120,180,240,300,360,480]
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0,0.0]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban","Semiurban","Rural"]
    )

    input_df = pd.DataFrame({

        "Gender":[gender],

        "Married":[married],

        "Dependents":[dependents],

        "Education":[education],

        "Self_Employed":[self_employed],

        "ApplicantIncome":[applicant_income],

        "CoapplicantIncome":[coapplicant_income],

        "LoanAmount":[loan_amount],

        "Loan_Amount_Term":[loan_term],

        "Credit_History":[credit_history],

        "Property_Area":[property_area]

    })

    return input_df


# ==========================================================
# Encode Prediction Data
# ==========================================================

def encode_prediction(input_df):

    df = input_df.copy()

    # Gender
    df["Gender"] = df["Gender"].map({
        "Male":1,
        "Female":0
    })

    # Married
    df["Married"] = df["Married"].map({
        "Yes":1,
        "No":0
    })

    # Dependents
    df["Dependents"] = df["Dependents"].replace({
        "0":0,
        "1":1,
        "2":2,
        "3+":3
    })

    # Education
    df["Education"] = df["Education"].map({
        "Graduate":1,
        "Not Graduate":0
    })

    # Self Employed
    df["Self_Employed"] = df["Self_Employed"].map({
        "Yes":1,
        "No":0
    })

    # Property Area
    df["Property_Area"] = df["Property_Area"].map({
        "Rural":0,
        "Semiurban":1,
        "Urban":2
    })

    return df


# ==========================================================
# Load Saved Model
# ==========================================================

def load_model():

    model = joblib.load(
        "saved_models/Best_Model.pkl"
    )

    return model


# ==========================================================
# Predict Loan Status
# ==========================================================

def predict_customer(model, input_df):

    prediction = model.predict(input_df)

    probability = None

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(
            input_df
        )

    return prediction, probability


# ==========================================================
# Show Prediction
# ==========================================================

def show_prediction(prediction):

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.success(
            "🎉 Loan Approved"
        )

    else:

        st.error(
            "❌ Loan Rejected"
        )


# ==========================================================
# Show Prediction Probability
# ==========================================================

def show_probability(probability):

    if probability is None:

        return

    st.subheader(
        "Prediction Probability"
    )

    approval = float(
        probability[0][1]
    )

    rejection = float(
        probability[0][0]
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Approval",
            f"{approval:.2%}"
        )

    with col2:

        st.metric(
            "Rejection",
            f"{rejection:.2%}"
        )

    st.progress(approval)


# ==========================================================
# Safe Prediction
# ==========================================================

def predict_with_error_handling(input_df):

    try:

        model = load_model()

        prediction, probability = predict_customer(

            model,

            input_df

        )

        show_prediction(
            prediction
        )

        show_probability(
            probability
        )

    except FileNotFoundError:

        st.error(

            "⚠ Train the model first."

        )

    except Exception as e:

        st.error(

            f"Prediction Error : {e}"

        )


