import streamlit as st


# def second_page():
model = st.selectbox("Select a model", [
    "",
    "Logistic Regression",
    "NeuralNetwork",
    "RandomForest Classification",
    "Naive Bayee Classifier",
    "GDBoost",
    "DecisionTree Classifier",
])

if model == "Logistic Regression":
    st.write(f"You selected {model}")
elif model == "NeuralNetwork":
    st.write(f"You selected {model}")
elif model == "RandomForest Classification":
    st.write(f"You selected {model}")
elif model == "Naive Bayee Classifier":
    st.write(f"You selected {model}")
elif model == "GDBoost":
    st.write(f"You selected {model}")
elif model == "DecisionTree Classifier":
    st.write(f"You selected {model}")
# if __name__ == "__main__":
#     second_page()
