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
    with open('logModelPre.pkl', 'rb') as f:
        logModelPre = pickle.load(f)
    st.text(logModelPre)

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
    with open('neuralNetworkPre.pkl', 'rb') as f:
        neuralNetworkPre = pickle.load(f)
    st.text(neuralNetworkPre)
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
    with open('randomForestPre.pkl', 'rb') as f:
        randomForestPre = pickle.load(f)
    st.text(randomForestPre)
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
    with open('naiveBayesPre.pkl', 'rb') as f:
        naiveBayesPre = pickle.load(f)
    st.text(naiveBayesPre)
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
    with open('gdBoostPre.pkl', 'rb') as f:
        gdBoostPre = pickle.load(f)
    st.text(gdBoostPre)
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
    with open('decTreePre.pkl', 'rb') as f:
        decTreePre = pickle.load(f)
    st.text(decTreePre)
