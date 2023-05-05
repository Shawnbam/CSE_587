import streamlit as st

import importlib.util

# Load the Python script
spec = importlib.util.spec_from_file_location("my_notebook", "Loan_Default.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Define input fields
input1 = st.text_input("Input 1")
input2 = st.slider("Input 2", 0, 10, 5)
input3 = st.selectbox("Input 3", ["Option 1", "Option 2", "Option 3"])

# Call the classification function
result = module.classify_data()

# Show the classification result
st.write("Classification result:", result)



# Define input fields
gender = st.selectbox("Gender", ["Male", "Female"])
loan_type = st.selectbox("Loan Type", ["Home loan", "Auto loan", "Personal loan"])
credit_worthiness = st.selectbox("Credit Worthiness", ["Good", "Fair", "Poor"])
open_credit = st.number_input("Open Credit", value=0)
business_or_commercial = st.selectbox("Business or Commercial", ["Yes", "No"])
loan_amount = st.number_input("Loan Amount", value=0)
interest_only = st.selectbox("Interest Only", ["Yes", "No"])
lump_sum_payment = st.selectbox("Lump Sum Payment", ["Yes", "No"])
construction_type = st.selectbox("Construction Type", ["Residential", "Commercial"])
occupancy_type = st.selectbox("Occupancy Type", ["Primary Residence", "Secondary Residence", "Investment Property"])
secured_by = st.selectbox("Secured By", ["Collateral", "Personal Guarantee"])
total_units = st.number_input("Total Units", value=0)
credit_type = st.selectbox("Credit Type", ["Fixed", "Variable"])
credit_score = st.number_input("Credit Score", value=0)
co_applicant_credit_type = st.selectbox("Co-Applicant Credit Type", ["Fixed", "Variable"])
region = st.selectbox("Region", ["East", "West", "North", "South"])
security_type = st.selectbox("Security Type", ["Mortgage", "Guarantee"])
status = st.number_input("Status", value=0)

# Show the input values
st.write("Gender:", gender)
st.write("Loan Type:", loan_type)
st.write("Credit Worthiness:", credit_worthiness)
st.write("Open Credit:", open_credit)
st.write("Business or Commercial:", business_or_commercial)
st.write("Loan Amount:", loan_amount)
st.write("Interest Only:", interest_only)
st.write("Lump Sum Payment:", lump_sum_payment)
st.write("Construction Type:", construction_type)
st.write("Occupancy Type:", occupancy_type)
st.write("Secured By:", secured_by)
st.write("Total Units:", total_units)
st.write("Credit Type:", credit_type)
st.write("Credit Score:", credit_score)
st.write("Co-Applicant Credit Type:", co_applicant_credit_type)
st.write("Region:", region)
st.write("Security Type:", security_type)
st.write("Status:", status)
