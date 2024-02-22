import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

def app():
    st.title("Iris Graph")

    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Create DataFrame
    iris_df = pd.DataFrame(X, columns=iris.feature_names)
    iris_df['species'] = y
    iris_df['species'] = iris_df['species'].map({i:target_names[i] for i in range(3)})

    # Plotting
    st.subheader("Histograms for each feature")
    for feature in iris.feature_names:
        st.write(f"## {feature}")
        sns.histplot(data=iris_df, x=feature, kde=True)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()