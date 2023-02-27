
import fastai
import streamlit as st
from fastai.vision.all import *
import pathlib
#import plotly.express as px

import pathlib
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

# Title
st.title('Ichimliklar klassifikatsiyasi!')

# Rasmni joylash
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg', 'jpg'])
if file:
    st.image(file)
    # PIL orqali konvert qilamiz
    img = PILImage.create(file)

    # Model
    model = load_learner('Ichimlik_model.pkl')

    # Prediction
    pred, pred_id, probs = model.predict(img)

    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimollik: {probs[pred_id] * 100:.2f} %')