import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pickle

from PIL import Image

st.markdown('# Predict if tumor is benign or malignant')
icon = Image.open("images/bandana.png")
st.image(icon)


st.title('This is a title')
st.header('Header')
st.subheader('Sub Header')
st.write('Can display many things')

#import time
#my_bar = st.progress(0)
#for percent_complete in range(100):
#    time.sleep(0.1)
#    my_bar.progress(percent_complete + 1)
#st.spinner()
#with st.spinner(text='In progress'):
#    time.sleep(5)
#    st.success('Done')
#e = RuntimeError('This is an exception of type RuntimeError')
#st.exception(e)

st.header('Header')

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

    result = f"Tumor is {output}."
    return result

features = []

features.append(st.number_input('Mean Radius',format="%.4f"))
features.append(st.number_input('Mean Texture',format="%.4f"))
features.append(st.number_input('Mean Perimeter',format="%.4f"))
features.append(st.number_input('Mean Area',format="%.4f"))
features.append(st.number_input('Mean Compactness',format="%.4f"))
features.append(st.number_input('Mean Concavity',format="%.4f"))
features.append(st.number_input('Mean Concave Points',format="%.4f"))
features.append(st.number_input('Radius Error',format="%.4f"))
features.append(st.number_input('Perimeter Error',format="%.4f"))
features.append(st.number_input('Area Error',format="%.4f"))
features.append(st.number_input('Concave Points Error',format="%.4f"))
features.append(st.number_input('Worst Radius',format="%.4f"))
features.append(st.number_input('Worst Texture',format="%.4f"))
features.append(st.number_input('Worst Perimeter',format="%.4f"))
features.append(st.number_input('Worst Area',format="%.4f"))
features.append(st.number_input('Worst Smoothness',format="%.4f"))
features.append(st.number_input('Worst Compactness',format="%.4f"))
features.append(st.number_input('Worst Concavity',format="%.4f"))
features.append(st.number_input('Worst Concave Points',format="%.4f"))
features.append(st.number_input('Worst Symmetry',format="%.4f"))

if st.button('Predict'):
    predictions = predict_tumor(features)
    st.write(predictions)



