import numpy as np
from numpy.testing._private.utils import measure
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pickle
from PIL import Image

def app():
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

    def measure_tumor():
        testset = pickle.load(open('testing_data.pkl', 'rb'))
        randsamp = testset.sample(n=1,replace=True)
        return randsamp.values.tolist()[0]

    col1, col2, col3, col4 = st.beta_columns((1,2,3,1))

    icon = Image.open("images/bandana.png")
    col2.image(icon, use_column_width=True)
    col3.title('Is your tumor benign or malignant?')
    col3.write('This application is created to help breast cancer patients and health care professionals to gain information on the tumor state using a machine learning model. It specifically uses the support vector machines algorithm to classify tumor state. Input the important metrics below to know whether the tumor is benign or malignant.')

    st.header('Click the button to measure tumor metrics:')

    features = []
    measurements = measure_tumor()

    meanradius = measurements[0]
    meantexture = measurements[1]
    meanperimeter = measurements[2]
    meanarea = measurements[3]
    meancompactness = measurements[4]
    meanconcavity = measurements[5]
    meanconcavepoints = measurements[6]
    radiuserror = measurements[7]
    perimetererror = measurements[8]
    areaerror = measurements[9]
    concavepointserror = measurements[10]
    worstradius = measurements[11]
    worsttexture = measurements[12]
    worstperimeter = measurements[13]
    worstarea = measurements[14]
    worstsmoothness = measurements[15]
    worstcompactness = measurements[16]
    worstconcavity = measurements[17]
    worstconcavepoints = measurements[18]
    worstsymmetry = measurements[19]

    mp = st.button('Measure & Predict')

    c1, c2, c3, c4 = st.beta_columns(4)

    features.append(c1.number_input('Mean Radius',format="%.4f",value=meanradius))
    features.append(c1.number_input('Mean Texture',format="%.4f",value=meantexture))
    features.append(c1.number_input('Mean Perimeter',format="%.4f",value=meanperimeter))
    features.append(c1.number_input('Mean Area',format="%.4f",value=meanarea))
    features.append(c1.number_input('Mean Compactness',format="%.4f",value=meancompactness))

    features.append(c2.number_input('Mean Concavity',format="%.4f",value=meanconcavity))
    features.append(c2.number_input('Mean Concave Points',format="%.4f",value=meanconcavepoints))
    features.append(c2.number_input('Radius Error',format="%.4f",value=radiuserror))
    features.append(c2.number_input('Perimeter Error',format="%.4f",value=perimetererror))
    features.append(c2.number_input('Area Error',format="%.4f",value=areaerror))

    features.append(c3.number_input('Concave Points Error',format="%.4f",value=concavepointserror))
    features.append(c3.number_input('Worst Radius',format="%.4f",value=worstradius))
    features.append(c3.number_input('Worst Texture',format="%.4f",value=worsttexture))
    features.append(c3.number_input('Worst Perimeter',format="%.4f",value=worstperimeter))
    features.append(c3.number_input('Worst Area',format="%.4f",value=worstarea))

    features.append(c4.number_input('Worst Smoothness',format="%.4f",value=worstsmoothness))
    features.append(c4.number_input('Worst Compactness',format="%.4f",value=worstcompactness))
    features.append(c4.number_input('Worst Concavity',format="%.4f",value=worstconcavity))
    features.append(c4.number_input('Worst Concave Points',format="%.4f",value=worstconcavepoints))
    features.append(c4.number_input('Worst Symmetry',format="%.4f",value=worstsymmetry))

    if mp:
        predictions = predict_tumor(features)
        st.write(predictions)
