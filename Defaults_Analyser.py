import streamlit as st
import pickles
import pandas as pd

markdown_text = """
# **Defaults Analyser**

Please fill the below form to identify if an user with below credentials would be a Loan Defaulter
"""

st.markdown(markdown_text, unsafe_allow_html=True)


loan_amount = st.text_input('loan_amount(1 = 100k)', 2.882519)
total_units = st.text_input('total_units', 4.000000)
credit_score = st.text_input('credit_score', 513.000000)
Gender_Female = st.selectbox('Gender_Female', ["No", "Yes"])
Gender_Joint = st.selectbox('Gender_Joint', ["No", "Yes"])
Gender_Male = st.selectbox('Gender_Male', ["No", "Yes"])
Gender_Sex = st.selectbox('Gender_Sex Not Available', ["Yes", "No"])
loan_type_type1 = st.selectbox('loan_type_type1', ["Yes", "No"])
loan_type_type2 = st.selectbox('loan_type_type2', ["No", "Yes"])
loan_type_type3 = st.selectbox('loan_type_type3', ["No", "Yes"])
Credit_Worthiness_l1 = st.selectbox('Credit_Worthiness_l1', ["Yes", "No"])
Credit_Worthiness_l2 = st.selectbox('Credit_Worthiness_l2', ["No", "Yes"])
open_credit_nopc = st.selectbox('open_credit_nopc', ["Yes", "No"])
open_credit_opc = st.selectbox('open_credit_opc', ["No", "Yes"])
business_or_commercial_b = st.selectbox(
    'business_or_commercial_b/c', ["No", "Yes"])
business_or_commercial_nob = st.selectbox(
    'business_or_commercial_nob/c', ["Yes", "No"])
interest_only_int_only = st.selectbox('interest_only_int_only', ["No", "Yes"])
interest_only_not_int = st.selectbox('interest_only_not_int', ["Yes", "No"])
lump_sum_payment_lpsm = st.selectbox('lump_sum_payment_lpsm', ["No", "Yes"])
lump_sum_payment_not_lpsm = st.selectbox(
    'lump_sum_payment_not_lpsm', ["Yes", "No"])
construction_type_mh = st.selectbox('construction_type_mh', ["No", "Yes"])
construction_type_sb = st.selectbox('construction_type_sb', ["Yes", "No"])
occupancy_type_ir = st.selectbox('occupancy_type_ir', ["Yes", "No"])
occupancy_type_pr = st.selectbox('occupancy_type_pr', ["No", "Yes"])
occupancy_type_sr = st.selectbox('occupancy_type_sr', ["No", "Yes"])
Secured_by_home = st.selectbox('Secured_by_home', ["Yes", "No"])
Secured_by_land = st.selectbox('Secured_by_land', ["No", "Yes"])
credit_type_CIB = st.selectbox('credit_type_CIB', ["Yes", "No"])
credit_type_CRIF = st.selectbox('credit_type_CRIF', ["No", "Yes"])
credit_type_EQUI = st.selectbox('credit_type_EQUI', ["No", "Yes"])
credit_type_EXP = st.selectbox('credit_type_EXP', ["No", "Yes"])
co_type_CIB = st.selectbox('co-applicant_credit_type_CIB', ["No", "Yes"])
co_type_EXP = st.selectbox('co-applicant_credit_type_EXP', ["Yes", "No"])
Region_central = st.selectbox('Region_central', ["No", "Yes"])
Region_north = st.selectbox('Region_north', ["No", "Yes"])
Region_north = st.selectbox('Region_north-east', ["No", "Yes"])
Region_south = st.selectbox('Region_south', ["Yes", "No"])
Security_Type_Indriect = st.selectbox('Security_Type_Indriect', ["No", "Yes"])
Security_Type_direct = st.selectbox('Security_Type_direct', ["Yes", "No"])
submit_button = st.button("Submit")

if submit_button:
    Gender_Female_val = float(0.00)
    if Gender_Female == "Yes":
        Gender_Female_val = float(1.00)
    Gender_Joint_val = float(0.00)
    if Gender_Joint == "Yes":
        Gender_Joint_val = float(1.00)
    Gender_Male_val = float(0.00)
    if Gender_Male == "Yes":
        Gender_Male_val = float(1.00)
    Gender_Sex_val = float(0.00)
    if Gender_Sex == "Yes":
        Gender_Sex_val = float(1.00)
    loan_type_type1_val = float(0.00)
    if loan_type_type1 == "Yes":
        loan_type_type1_val = float(1.00)
    loan_type_type2_val = float(0.00)
    if loan_type_type2 == "Yes":
        loan_type_type2_val = float(1.00)
    loan_type_type3_val = float(0.00)
    if loan_type_type3 == "Yes":
        loan_type_type3_val = float(1.00)
    Credit_Worthiness_l1_val = float(0.00)
    if Credit_Worthiness_l1 == "Yes":
        Credit_Worthiness_l1_val = float(1.00)
    Credit_Worthiness_l2_val = float(0.00)
    if Credit_Worthiness_l2 == "Yes":
        Credit_Worthiness_l2_val = float(1.00)
    open_credit_nopc_val = float(0.00)
    if open_credit_nopc == "Yes":
        open_credit_nopc_val = float(1.00)
    open_credit_opc_val = float(0.00)
    if open_credit_opc == "Yes":
        open_credit_opc_val = float(1.00)
    business_or_commercial_b_val = float(0.00)
    if business_or_commercial_b == "Yes":
        business_or_commercial_b_val = float(1.00)
    business_or_commercial_nob_val = float(0.00)
    if business_or_commercial_nob == "Yes":
        business_or_commercial_nob_val = float(1.00)
    interest_only_int_only_val = float(0.00)
    if interest_only_int_only == "Yes":
        interest_only_int_only_val = float(1.00)
    interest_only_not_int_val = float(0.00)
    if interest_only_not_int == "Yes":
        interest_only_not_int_val = float(1.00)
    lump_sum_payment_lpsm_val = float(0.00)
    if lump_sum_payment_lpsm == "Yes":
        lump_sum_payment_lpsm_val = float(1.00)
    lump_sum_payment_not_lpsm_val = float(0.00)
    if lump_sum_payment_not_lpsm == "Yes":
        lump_sum_payment_not_lpsm_val = float(1.00)
    construction_type_mh_val = float(0.00)
    if construction_type_mh == "Yes":
        construction_type_mh_val = float(1.00)
    construction_type_sb_val = float(0.00)
    if construction_type_sb == "Yes":
        construction_type_sb_val = float(1.00)
    occupancy_type_ir_val = float(0.00)
    if occupancy_type_ir == "Yes":
        occupancy_type_ir_val = float(1.00)
    occupancy_type_pr_val = float(0.00)
    if occupancy_type_pr == "Yes":
        occupancy_type_pr_val = float(1.00)
    occupancy_type_sr_val = float(0.00)
    if occupancy_type_sr == "Yes":
        occupancy_type_sr_val = float(1.00)
    Secured_by_home_val = float(0.00)
    if Secured_by_home == "Yes":
        Secured_by_home_val = float(1.00)
    Secured_by_land_val = float(0.00)
    if Secured_by_land == "Yes":
        Secured_by_land_val = float(1.00)
    credit_type_CIB_val = float(0.00)
    if credit_type_CIB == "Yes":
        credit_type_CIB_val = float(1.00)
    credit_type_CRIF_val = float(0.00)
    if credit_type_CRIF == "Yes":
        credit_type_CRIF_val = float(1.00)
    credit_type_EQUI_val = float(0.00)
    if credit_type_EQUI == "Yes":
        credit_type_EQUI_val = float(1.00)
    credit_type_EXP_val = float(0.00)
    if credit_type_EXP == "Yes":
        credit_type_EXP_val = float(1.00)
    co_type_CIB_val = float(0.00)
    if co_type_CIB == "Yes":
        co_type_CIB_val = float(1.00)
    co_type_EXP_val = float(0.00)
    if co_type_EXP == "Yes":
        co_type_EXP_val = float(1.00)
    Region_central_val = float(0.00)
    if Region_central == "Yes":
        Region_central_val = float(1.00)
    Region_north_val = float(0.00)
    if Region_north == "Yes":
        Region_north_val = float(1.00)
    Region_north_val = float(0.00)
    if Region_north == "Yes":
        Region_north_val = float(1.00)
    Region_south_val = float(0.00)
    if Region_south == "Yes":
        Region_south_val = float(1.00)
    Security_Type_Indriect_val = float(0.00)
    if Security_Type_Indriect == "Yes":
        Security_Type_Indriect_val = float(1.00)
    Security_Type_direct_val = float(0.00)
    if Security_Type_direct == "Yes":
        Security_Type_direct_val = float(1.00)
    val = [
        float(loan_amount),
        float(total_units),
        float(credit_score),
        Gender_Female_val,
        Gender_Joint_val,
        Gender_Male_val,
        Gender_Sex_val,
        loan_type_type1_val,
        loan_type_type2_val,
        loan_type_type3_val,
        Credit_Worthiness_l1_val,
        Credit_Worthiness_l2_val,
        open_credit_nopc_val,
        open_credit_opc_val,
        business_or_commercial_b_val,
        business_or_commercial_nob_val,
        interest_only_int_only_val,
        interest_only_not_int_val,
        lump_sum_payment_lpsm_val,
        lump_sum_payment_not_lpsm_val,
        construction_type_mh_val,
        construction_type_sb_val,
        occupancy_type_ir_val,
        occupancy_type_pr_val,
        occupancy_type_sr_val,
        Secured_by_home_val,
        Secured_by_land_val,
        credit_type_CIB_val,
        credit_type_CRIF_val,
        credit_type_EQUI_val,
        credit_type_EXP_val,
        co_type_CIB_val,
        co_type_EXP_val,
        Region_central_val,
        Region_north_val,
        Region_north_val,
        Region_south_val,
        Security_Type_Indriect_val,
        Security_Type_direct_val,
    ]
    print(val)
    result = pickles.getRandomForestClassification([val])
    predictions = [int(p) for p in result]

    # for p in range(len(predictions)):
    #     print("result is", predictions[p])
    #     if predictions[p] == 0:
    #         status = st.text_input("Status " + "getRandomForestClassification", value="Loan Defaulter")
    #     else:
    #         status = st.text_input(
    #             "Status " + "getRandomForestClassification", value="Not a Loan Defaulter")

    # result = pickles.getLogisticClassification([val])
    # predictions = [int(p) for p in result]

    # for p in range(len(predictions)):
    #     print("result is", predictions[p])
    #     if predictions[p] == 0:
    #         status = st.text_input("Status " + "getLogisticClassification", value="Loan Defaulter")
    #     else:
    #         status = st.text_input(
    #             "Status " + "getLogisticClassification", value="Not a Loan Defaulter")

    # result = pickles.getCLFClassification([val])
    # predictions = [int(p) for p in result]

    # for p in range(len(predictions)):
    #     print("result is", predictions[p])
    #     if predictions[p] == 0:
    #         status = st.text_input("Status " + "getCLFClassification", value="Loan Defaulter")
    #     else:
    #         status = st.text_input(
    #             "Status " + "getCLFClassification", value="Not a Loan Defaulter")

    result = pickles.getNBClassification([val])
    predictions = [int(p) for p in result]

    for p in range(len(predictions)):
        print("result is", predictions[p])
        if predictions[p] == 0:
            status = st.text_input(
                "Status " + "getNBClassification", value="Loan Defaulter")
        else:
            status = st.text_input(
                "Status " + "getNBClassification", value="Not a Loan Defaulter")

    # result = pickles.getGDBoostClassification([val])
    # predictions = [int(p) for p in result]

    # for p in range(len(predictions)):
    #     print("result is", predictions[p])
    #     if predictions[p] == 0:
    #         status = st.text_input("Status " + "getGDBoostClassification", value="Loan Defaulter")
    #     else:
    #         status = st.text_input(
    #             "Status " + "getGDBoostClassification", value="Not a Loan Defaulter")


# [0.779293, 1.000000, 813.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000]
