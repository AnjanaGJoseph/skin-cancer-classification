import streamlit as st 
import pickle
from PIL import Image as PILimage
from fastai.vision import *
from fastai.callbacks.hooks import *
import numpy as np
st.title("Skin Disease Prediction")
st.subheader("Upload an image of your skin to check the kind of disease")
st.spinner("Testing spinner")

uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))

if uploaded_file is not None:
    image = PILimage.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    
    if st.button('Check Now'):
        st.write("Classifying...")
        learner = load_learner('./')

        pred_class,pred_idx,outputs = learner.predict(image)
        if(pred_class == tensor(0)):
            st.write("Actinic Keratoses")
        elif(pred_class == tensor(1)):
            st.write("Basal cell carcinoma")
        elif(pred_class == tensor(2)): 
            st.write("Benign keratosis")
        
        elif(pred_class == tensor(3)):
            st.write("Dermatofibroma")
        elif(pred_class == tensor(4)):
            st.write("Melanoma")
        elif(pred_class == tensor(5)):
            st.write("Melanocytic nevi ")
        elif(pred_class == tensor(6)):
            st.write("Vascular skin lesions ")  
