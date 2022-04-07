import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'SepalLengthCm': sepal_length,
            'SepalWidthCm': sepal_width,
            'PetalLengthCm': petal_length,
            'PetalWidthCm': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = pd.read_csv("https://raw.githubusercontent.com/aaa-yanbin/might-march-streamlit/main/Iris.csv")
iris.Species = iris.Species.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
X = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
Y = iris['Species']

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

target_names = pd.Series(['setosa', 'versicolor', 'virginica'])

st.subheader('Class labels and their corresponding index number')
st.write(target_names)

st.subheader('Prediction')
st.write(target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
