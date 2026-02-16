import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Load model
# with open("final_model.pkl", "rb") as f:
#     model = pickle.load(f)

# # Load scaler
# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# Load columns
# with open("columns.json", "r") as f:
#     columns = json.load(f)

# =========================================
# Load Model + Scaler
# =========================================
model = pickle.load(open("final_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Loan EMI Prediction Dashboard")

st.write("Enter customer details to predict Maximum EMI")

# =========================================
# User Inputs
# =========================================

age = st.number_input("Age", 18, 60)

gender = st.selectbox("Gender", ["Male", "Female"])
gender = 0 if gender == "Male" else 1

marital_status = st.selectbox("Marital Status", ["Single", "Married"])
marital_status = 0 if marital_status == "Single" else 1

education = st.selectbox(
    "Education",
    ["High School", "Graduate", "Post Graduate", "Professional"]
)

education_map = {
    "High School": 1,
    "Graduate": 2,
    "Post Graduate": 3,
    "Professional": 4
}
education = education_map[education]

monthly_salary = st.number_input("Monthly Salary")

employment_type = st.selectbox(
    "Employment Type",
    ["Private", "Government", "Self-employed"]
)

emp_map = {
    "Private": 0,
    "Government": 1,
    "Self-employed": 2
}
employment_type = emp_map[employment_type]

years_of_employment = st.number_input("Years of Employment")

company_type = st.selectbox(
    "Company Type",
    ["Mid-size", "MNC", "Startup", "Large Indian", "Small"]
)

company_map = {
    "Mid-size": 0,
    "MNC": 1,
    "Startup": 2,
    "Large Indian": 3,
    "Small": 4
}
company_type = company_map[company_type]

house_type = st.selectbox(
    "House Type",
    ["Own", "Rented", "Family"]
)

house_map = {
    "Own": 0,
    "Rented": 1,
    "Family": 2
}
house_type = house_map[house_type]

monthly_rent = st.number_input("Monthly Rent")

family_size = st.number_input("Family Size")

dependents = st.number_input("Dependents")

school_fees = st.number_input("School Fees")

college_fees = st.number_input("College Fees")

travel_expenses = st.number_input("Travel Expenses")

groceries_utilities = st.number_input("Groceries & Utilities")

other_monthly_expenses = st.number_input("Other Monthly Expenses")

existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
existing_loans = 1 if existing_loans == "Yes" else 0

current_emi_amount = st.number_input("Current EMI Amount")

credit_score = st.number_input("Credit Score")

bank_balance = st.number_input("Bank Balance")

emergency_fund = st.number_input("Emergency Fund")

emi_scenario = st.selectbox(
    "EMI Scenario",
    [
        "Personal Loan EMI",
        "E-commerce Shopping EMI",
        "Education EMI",
        "Vehicle EMI",
        "Home Appliances EMI"
    ]
)

emi_map = {
    "Personal Loan EMI": 0,
    "E-commerce Shopping EMI": 1,
    "Education EMI": 2,
    "Vehicle EMI": 3,
    "Home Appliances EMI": 4
}
emi_scenario = emi_map[emi_scenario]

requested_amount = st.number_input("Requested Loan Amount")

requested_tenure = st.number_input("Requested Tenure (Months)")

# =========================================
# Predict Button
# =========================================

if st.button("Predict Maximum EMI"):

    input_df = pd.DataFrame([[
        age, gender, marital_status, education,
        monthly_salary, employment_type,
        years_of_employment, company_type,
        house_type, monthly_rent,
        family_size, dependents,
        school_fees, college_fees,
        travel_expenses, groceries_utilities,
        other_monthly_expenses, existing_loans,
        current_emi_amount, credit_score,
        bank_balance, emergency_fund,
        emi_scenario, requested_amount,
        requested_tenure
    ]],
    columns=[
        'age','gender','marital_status','education',
        'monthly_salary','employment_type',
        'years_of_employment','company_type',
        'house_type','monthly_rent',
        'family_size','dependents',
        'school_fees','college_fees',
        'travel_expenses','groceries_utilities',
        'other_monthly_expenses','existing_loans',
        'current_emi_amount','credit_score',
        'bank_balance','emergency_fund',
        'emi_scenario','requested_amount',
        'requested_tenure'
    ])

    # =========================
    # Apply Scaling
    # =========================
    num_cols = [
        'monthly_salary',
        'years_of_employment',
        'monthly_rent',
        'current_emi_amount',
        'credit_score',
        'bank_balance',
        'requested_amount'
    ]

    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # =========================
    # Prediction
    # =========================
    prediction = model.predict(input_df)

    st.success(f"Maximum Eligible EMI: ₹ {prediction[0]:,.2f}")

