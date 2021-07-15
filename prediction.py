#====================================================================================
# Python prediction file - contains predict function and associated helper functions
#====================================================================================
# Users can use the pre-trained model, or create a new model for predicted repairs
#
#====================================================================================

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

#==========================================================
# Build dataset - use below to create train & test datasets
# Use Use 80% of data for training, 20% for testing.
# max_words is the max amoutn of words in user's text input.
#=========================================================

training_portion = .80  
max_words = 1000        

data = pd.read_csv('dataset/testdata1.csv')
train_size = int(len(data) * training_portion)

#==========================================================
# Split the data for testing and training
#==========================================================
def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test

train_cat, test_cat = train_test_split(data.iloc[:,1], train_size)  # label data is second column
train_text, test_text = train_test_split(data.iloc[:,0], train_size)  # text data is first column

tokenize = Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_text) # fit tokenizer to our training text data

#x_train and x_test are the vectorization of the text data (which is a claim)
x_train = tokenize.texts_to_matrix(train_text)
x_test = tokenize.texts_to_matrix(test_text)

#==========================================================
# Use sklearn to convert label strings to a numbered index
#==========================================================
encoder = LabelEncoder()  
encoder.fit(train_cat)

#==========================================================
# Convert label strings to numbers
#==========================================================
y_train = encoder.transform(train_cat)
y_test = encoder.transform(test_cat)

#==========================================================
# Convert the labels to a one-hot representation
#==========================================================
num_classes = len(set(y_train))  
y_train = to_categorical(y_train, num_classes)  
y_test = to_categorical(y_test, num_classes)

#==========================================================
# Build the model
#==========================================================
layers = keras.layers
models = keras.models
model = models.Sequential()
model.add(layers.Dense(512, input_shape=(max_words,), activation='relu'))  # Hidden layer with 512 nodes
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
batch_size = 32
epochs = 2

history = model.fit(x_train, y_train,       
                    batch_size=batch_size,  
                    epochs=epochs,          
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,       
                       batch_size=batch_size, verbose=1)

#==========================================================
# text_abels is an ndarray of output values (labels or 
# classes)  e.g. other, brakes, starter
#==========================================================
text_labels = encoder.classes_   

#==========================================================
# Save labels (categories) to be used later when we run the
# model in flask app
#==========================================================
#with open('dataset/savedCategories.csv', 'w') as file:
#    writer = csv.writer(file, delimiter=',', lineterminator='\\n')
#    for i in range(len(text_labels)):
#        temp = text_labels[i]
#        writer.writerow(temp)


#==========================================================
# TRY - examine the first 10 test samples of 445
#==========================================================
#for i in range(len(test_cat)):
#    temp = x_test[i]
#    prediction = model.predict(np.array([x_test[i]]))
#    predicted_label = text_labels[np.argmax(prediction)]  #predicted class
 
#==========================================================
# SAVE model - save the model as an hf file
#==========================================================
# model.save('models/repairmodel.h5')  #after prediction, save the model  
def save_model():
    model.save('models/repairmodel.h5')
    return

#==========================================================
# LOAD model - load model from saved hf file
#==========================================================
# model = keras.models.load_model('models/repairmodel.h5')
def load_model():
    model = keras.models.load_model('models/repairmodel.h5')
    return

#==========================================================
#Prediction function - takes input of text describing a
#repair issue.  e.g. 'when I turn the key I hear a clicking noise'
#==========================================================
def predict(single_test_text):
    model = keras.models.load_model('models/repairmodel.h5')
    text_as_series = pd.Series(single_test_text) #do a data conversion
    single_x_test = tokenize.texts_to_matrix(text_as_series)
    single_prediction = model.predict(np.array([single_x_test]))
    model.save('models/repairmodel.h5')  #after prediction, save the model
    single_predicted_label = text_labels[np.argmax(single_prediction)]  #maps index of the prediction to the test labels array e.g. brakes
    return {'prediction': single_predicted_label}

#==========================================================
# test prediction function 
# prediction = predict(single_test_text)  
# print('returned prediction: ' + prediction)
#==========================================================
