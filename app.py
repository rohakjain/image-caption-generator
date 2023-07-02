# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 13:16:10 2023

@author: rjroh
"""

###################    imports
import PIL
from PIL import Image
import numpy as np
import pickle
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

import streamlit as st
###################

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

# generate captions for an image
def predict_caption(model, image, tokenizer, max_length):
    #add start tag for generation process
    in_text='startseq'
    #iterate over the max length sequence
    for i in range(max_length):
        #encode input sequence 
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        #predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        #convert index to word
        word = idx_to_word(yhat, tokenizer)
        #stop if word not found
        if word is None: 
            break
        # append word as input for generating next word
        in_text += " " + word
        #stop if we reach end tag
        if word=='endseq':
            break
    return in_text

def predictor_func(uploaded_file):
    
    # load the image from file
    image = Image.open(uploaded_file)
    image = image.resize((224,224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)
    # extract features
    features = model.predict(image, verbose=0)
    tokenizer = Tokenizer()
    with open("./tokenizer.pkl", 'rb') as f:
        tokenizer = pickle.load(f)
    final_model = load_model("./best_model_final.h5",compile=False)
    
    y_pred = predict_caption(final_model, features, tokenizer, 35)
    return y_pred



def main():
    # Load vgg16 Model
    model = VGG16()
    # restructure model
    model = Model(inputs = model.inputs , outputs = model.layers[-2].output)
    st.title("Image caption generator")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        caption = ''
        if st.button('Generate caption'):
            caption=predictor_func(uploaded_file)
        st.success(caption)
    
if __name__ == "__main__":
    main()