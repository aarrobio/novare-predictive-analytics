import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pickle
from PIL import Image

st.set_page_config(page_title='Know Your Tumor State',layout="wide")

col1, col2, col3, col4 = st.beta_columns((1,2,3,1))

icon = Image.open("images/bandana.png")
col2.image(icon, use_column_width=True)
col3.title('Is your tumor benign or malignant?')
col3.write('This application is created to help breast cancer patients and health care professionals to gain information on the tumor state using a machine learning model. It specifically uses the support vector machines algorithm to classify tumor state. Input the important metrics below to know whether the tumor is benign or malignant.')

st.header('Input metrics below:')

def predict_tumor(features):
    model = pickle.load(open('svc_fin.pkl', 'rb'))
    trainset = pickle.load(open('training_data.pkl', 'rb'))
    scaler = StandardScaler()
    
    #newdata = pd.Series(features, index=trainset.columns)
    #trainset = trainset.append(newdata, ignore_index=True)
    newdata = pd.DataFrame([features], columns=trainset.columns)
    trainset = scaler.fit_transform(trainset)

    final_features = scaler.transform(newdata)
    prediction = model.predict(final_features)

    output = prediction[0]

    result = f"Your tumor is {output}."
    return result

features = []

c1, c2, c3, c4 = st.beta_columns(4)
features.append(c1.number_input('Mean Radius',format="%.4f",value=17.99))
features.append(c1.number_input('Mean Texture',format="%.4f",value=10.38))
features.append(c1.number_input('Mean Perimeter',format="%.4f",value=122.80))
features.append(c1.number_input('Mean Area',format="%.4f",value=1001.0))
features.append(c1.number_input('Mean Compactness',format="%.4f",value=0.27760))

features.append(c2.number_input('Mean Concavity',format="%.4f",value=0.30010))
features.append(c2.number_input('Mean Concave Points',format="%.4f",value=0.14710))
features.append(c2.number_input('Radius Error',format="%.4f",value=1.0950))
features.append(c2.number_input('Perimeter Error',format="%.4f",value=8.589))
features.append(c2.number_input('Area Error',format="%.4f",value=153.40))

features.append(c3.number_input('Concave Points Error',format="%.4f",value=0.01587))
features.append(c3.number_input('Worst Radius',format="%.4f",value=25.38))
features.append(c3.number_input('Worst Texture',format="%.4f",value=17.33))
features.append(c3.number_input('Worst Perimeter',format="%.4f",value=184.60))
features.append(c3.number_input('Worst Area',format="%.4f",value=2019.0	))

features.append(c4.number_input('Worst Smoothness',format="%.4f",value=0.1622))
features.append(c4.number_input('Worst Compactness',format="%.4f",value=0.6656))
features.append(c4.number_input('Worst Concavity',format="%.4f",value=0.7119))
features.append(c4.number_input('Worst Concave Points',format="%.4f",value=0.2654))
features.append(c4.number_input('Worst Symmetry',format="%.4f",value=0.4601))

if st.button('Predict'):
    predictions = predict_tumor(features)
    st.write(predictions)

