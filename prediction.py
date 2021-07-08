#===================================================
#In this script labels are created for our test data, convert the labels to numbered indexes and then use one-hot encoding.  

#-- One hot encoding allows the representation of categorical data to be more expressive. Many machine learning algorithms cannot work with categorical data directly. The categories must be converted into numbers. This is required for both input and output variables that are categorical.

#--After we have converted the labels using one-hot encoding, we are ready to build our main NLP model and train it.

#--Once the model is trained, we can test our model by entering a claim (e.g. the brakes feel soft when I press on them) and check if the model has correctly characterized the claim.  --#

# Load your model.
# model_dir = 'models/myfancymodel'
# saved_model = tf.saved_model.load(model_dir)
# predictor = saved_model.signatures['default']

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

training_portion = .80  # Use 80% of data for training, 20% for testing
max_words = 1000        #Max words in text input

data = pd.read_csv('testdata1.csv')
train_size = int(len(data) * training_portion)

#==========================================================
#split the data for testing and training
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
#convert label strings to numbers
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

text_labels = encoder.classes_   #ndarray of output values (labels or classes)  e.g. other, brakes, starter

#==========================================================
# Examine first 10 test samples of 445
#==========================================================
for i in range(len(test_cat)):
    temp = x_test[i]
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]  #predicted class

#==========================================================
#Prediction function - takes input of text describing a
#repair issue.  e.g. 'when I turn the key I hear a clicking noise'
#==========================================================
def predict(single_test_text):
    text_as_series = pd.Series(single_test_text) #do a data conversion
    single_x_test = tokenize.texts_to_matrix(text_as_series)
    single_prediction = model.predict(np.array([single_x_test]))
    single_predicted_label = text_labels[np.argmax(single_prediction)]  #maps index of the prediction to the test labels array e.g. brakes
    # return (single_predicted_label)
    return {'prediction': single_predicted_label}
#test prediction function ===========================================
#prediction = predict(single_test_text)  
#print('returned prediction: ' + prediction)
