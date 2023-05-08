import streamlit as st
import seaborn as sns
# from Loan_Default import logisticRegressionConfusionMatrix, decisionTreeConfusionMatrix, gdBoostConfusionMatrix, naiveBayesConfusionMatrix, randomForestConfusionMatrix, neuralNetworkConfusionMatrix
import matplotlib.pyplot as plt
import io
import pickle

if 'my_key' not in st.session_state:
    st.session_state['my_key'] = None

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
    logisticRegressionConfusionMatrix = None
    with open('logisticRegressionConfusionMatrix.pkl', 'rb') as f:
        logisticRegressionConfusionMatrix = pickle.load(f)
    sns.heatmap(logisticRegressionConfusionMatrix, annot=True, cmap='Greens')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Get the Matplotlib figure object
    fig = plt.gcf()

    # Display the plot in Streamlit
    st.pyplot(fig)

elif model == "NeuralNetwork":
    st.write(f"You selected {model}")

    neuralNetworkConfusionMatrix = None
    with open('neuralNetworkConfusionMatrix.pkl', 'rb') as f:
        neuralNetworkConfusionMatrix = pickle.load(f)
    sns.heatmap(neuralNetworkConfusionMatrix, annot=True, cmap='Greens')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Get the Matplotlib figure object
    fig = plt.gcf()

    # Display the plot in Streamlit
    st.pyplot(fig)
elif model == "RandomForest Classification":
    st.write(f"You selected {model}")

    randomForestConfusionMatrix = None
    with open('randomForestConfusionMatrix.pkl', 'rb') as f:
        randomForestConfusionMatrix = pickle.load(f)
    sns.heatmap(randomForestConfusionMatrix, annot=True, cmap='Greens')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Get the Matplotlib figure object
    fig = plt.gcf()

    # Display the plot in Streamlit
    st.pyplot(fig)
elif model == "Naive Bayee Classifier":
    st.write(f"You selected {model}")

    naiveBayesConfusionMatrix = None
    with open('naiveBayesConfusionMatrix.pkl', 'rb') as f:
        naiveBayesConfusionMatrix = pickle.load(f)
    sns.heatmap(naiveBayesConfusionMatrix, annot=True, cmap='Greens')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Get the Matplotlib figure object
    fig = plt.gcf()

    # Display the plot in Streamlit
    st.pyplot(fig)
elif model == "GDBoost":
    st.write(f"You selected {model}")

    gdBoostConfusionMatrix = None
    with open('gdBoostConfusionMatrix.pkl', 'rb') as f:
        gdBoostConfusionMatrix = pickle.load(f)
    sns.heatmap(gdBoostConfusionMatrix, annot=True, cmap='Greens')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Get the Matplotlib figure object
    fig = plt.gcf()

    # Display the plot in Streamlit
    st.pyplot(fig)
elif model == "DecisionTree Classifier":
    st.write(f"You selected {model}")

    decisionTreeConfusionMatrix = None
    with open('decisionTreeConfusionMatrix.pkl', 'rb') as f:
        decisionTreeConfusionMatrix = pickle.load(f)
    sns.heatmap(decisionTreeConfusionMatrix, annot=True, cmap='Greens')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Get the Matplotlib figure object
    fig = plt.gcf()

    # Display the plot in Streamlit
    st.pyplot(fig)
