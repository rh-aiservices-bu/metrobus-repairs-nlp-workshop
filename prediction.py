from tensorflow import keras
import pandas as pd
import numpy as np
import pickle

#load model
model_dir = 'models/repairmodel.h5'
model = keras.models.load_model(model_dir)

#load labels
with open('text_labels.npy', 'rb') as f:
    text_labels = np.load(f, allow_pickle=True)

#load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenize = pickle.load(handle)



def predict(single_test_text):

    text_as_series = pd.Series(single_test_text) #do a data conversion
    single_x_test = tokenize.texts_to_matrix(text_as_series)
    single_prediction = model.predict(np.array([single_x_test]))

    single_predicted_label = text_labels[np.argmax(single_prediction)]
    
    return {'prediction': single_predicted_label}
