import streamlit as st

# Create input fields
input1 = st.text_input("Input 1")
input2 = st.slider("Input 2", 0, 10, 5)
input3 = st.selectbox("Input 3", ["Option 1", "Option 2", "Option 3"])
input4 = st.checkbox("Input 4")
input5 = st.date_input("Input 5")
input6 = st.file_uploader("Input 6")

# Show inputs
st.write("Input 1:", input1)
st.write("Input 2:", input2)
st.write("Input 3:", input3)
st.write("Input 4:", input4)
st.write("Input 5:", input5)
st.write("Input 6:", input6)
