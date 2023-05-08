import streamlit as st

def other_page():
    number = st.selectbox("Select a number", range(1, 7))
    st.write(f"You selected {number}")

if __name__ == "__main__":
    other_page()
