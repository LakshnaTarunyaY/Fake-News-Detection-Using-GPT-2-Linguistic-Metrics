import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from model import load_model
from model import extract

st.title("Fake News Detection using LLM Metrics")
st.write("Upload a dataset containing news text and labels.")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df=df.dropna(subset=['text'])
    st.write("Dataset Preview")
    st.write(df.head(100))

    @st.cache_resource
    def get_model():
        return load_model()
    tokenizer, model = get_model()
    texts=df['text']
    labels=df['subject']
    st.write('Extracting features')

    features,perplexities,accuracies=extract(texts,tokenizer,model)
    x=np.array(features)
    y=labels.values

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    m=LogisticRegression()
    m.fit(x_train,y_train)
    predictions=m.predict(x_test)
    accuracy=accuracy_score(y_test,predictions)
    st.subheader("Model Accuracy")
    st.write(accuracy)
    
    # Perplexity Visualization
    st.subheader("Perplexity Distribution")
    fig1, ax1 = plt.subplots()
    ax1.hist(perplexities, bins=20)
    ax1.set_title("Perplexity Distribution")
    ax1.set_xlabel("Perplexity")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

    # Token Accuracy Visualization
    st.subheader("Token Prediction Accuracy Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(accuracies, bins=20)
    ax2.set_title("Token Prediction Accuracy")
    ax2.set_xlabel("Accuracy")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)
    
    # Feature Scatter Plot
    st.subheader("Perplexity vs Token Accuracy")
    fig3, ax3 = plt.subplots()
    ax3.scatter(perplexities, accuracies)
    ax3.set_xlabel("Perplexity")
    ax3.set_ylabel("Token Accuracy")
    ax3.set_title("Feature Relationship")
    st.pyplot(fig3)


