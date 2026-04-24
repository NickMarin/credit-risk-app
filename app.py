import streamlit as st
import pickle
import numpy as np

# Load model
with open('credit_risk_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Credit Risk Assessor", page_icon="🏦", layout="centered")

st.title("🏦 Credit Risk Assessor")
st.markdown("Enter the customer's financial details to assess credit risk.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    monthly_income = st.number_input("Monthly Income (€)", 0, 50000, 3000)
    debt_ratio = st.slider("Debt Ratio", 0.0, 1.0, 0.3)
    revolving_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.3)

with col2:
    late_30_59 = st.slider("Times 30-59 Days Late", 0, 10, 0)
    late_60_89 = st.slider("Times 60-89 Days Late", 0, 10, 0)
    late_90 = st.slider("Times 90+ Days Late", 0, 10, 0)
    dependents = st.slider("Number of Dependents", 0, 10, 0)
    open_credit_lines = st.slider("Open Credit Lines", 0, 30, 5)
    real_estate_loans = st.slider("Real Estate Loans", 0, 10, 1)

# Engineered features
total_late = late_30_59 + late_60_89 + late_90
income_per_dependent = monthly_income / (dependents + 1)
debt_to_income = debt_ratio * monthly_income
zero_income = 1 if monthly_income == 0 else 0

features = np.array([[
    revolving_utilization, age, late_30_59, debt_ratio,
    monthly_income, open_credit_lines, late_90,
    real_estate_loans, late_60_89, dependents,
    total_late, income_per_dependent, debt_to_income, zero_income
]])

if st.button("🔍 Assess Credit Risk", use_container_width=True):
    prob = model.predict_proba(features)[0][1]
    
    st.divider()
    
    if prob < 0.15:
        st.success(f"✅ ACCEPT — Low Risk Customer")
    elif prob < 0.30:
        st.warning(f"⚠️ REVIEW — Medium Risk Customer")
    else:
        st.error(f"❌ REJECT — High Risk Customer")
    
    st.metric("Default Probability", f"{prob:.1%}")
    
    # Risk factors
    st.subheader("Key Risk Factors")
    if total_late > 0:
        st.write(f"⚠️ Customer has {total_late} late payment(s) on record")
    if revolving_utilization > 0.7:
        st.write("⚠️ High credit utilization")
    if debt_ratio > 0.5:
        st.write("⚠️ High debt ratio")
    if monthly_income < 2000:
        st.write("⚠️ Low monthly income")
    if total_late == 0 and revolving_utilization < 0.3:
        st.write("✅ Clean payment history and low credit utilization")
