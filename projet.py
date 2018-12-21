#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 20:39:14 2018

@author: estelle
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast
import os
from keras.layers import Dense, Embedding, Input
from keras.layers import GRU, Dropout, MaxPooling1D, Conv1D, Flatten, LSTM
from keras.models import Model
import numpy as np
import itertools
from keras.utils import np_utils
from sklearn.metrics import (classification_report, 
                             precision_recall_fscore_support, 
                             accuracy_score)

from keras.preprocessing import text, sequence
from sklearn import preprocessing


# Chargement du fichier text (affichage des 10 premier résultats)
df=pd.read_csv("data/tobacco-lab_data_Tobacco3482.csv")

# On extrait le texte des fichiers et l'on stocke chaque texte dans une liste text
# On stocke les label de chaque text dans une liste label
label=[]
texte=[]
for i in range(df.shape[0]):
    path=path='data/Tobacco3482-OCR/'+str(df.img_path[i][:-3])+'txt'
    with open(path) as myfile:
        content = myfile.readlines()
        for j,e in enumerate(content):
            #retrait des /n pour une meilleurs lisibilité et augmente les performances
            content[j]=e.rstrip("\n")
    texte.append(content)
    label.append(df.label[i])
    
    
# création d'un dataframe contenant chaque test et les labels associés
cont=pd.DataFrame({'text':texte, 'label':label})

cont['text']=[" ".join(texte) for texte in cont['text'].values]


# Model parameters
MAX_FEATURES = 2000
MAX_TEXT_LENGTH = 2000
EMBED_SIZE  = 100
BATCH_SIZE = 16
EPOCHS = 30
VALIDATION_SPLIT = 0.1


#séparation en ensemble test apprentissage
X_train,X_test, y_train,y_test = train_test_split(cont.text, cont.label, test_size=0.10, 
                                                random_state=42)
print('nb exemple apprentissage :' ,X_train.shape)
print('nb exemple de test:', X_test.shape)

def get_train_test(train_raw_text, test_raw_text):
    
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)

    tokenizer.fit_on_texts(list(train_raw_text))
    train_tokenized = tokenizer.texts_to_sequences(train_raw_text)
    test_tokenized = tokenizer.texts_to_sequences(test_raw_text)
    return sequence.pad_sequences(train_tokenized, maxlen=MAX_TEXT_LENGTH), \
           sequence.pad_sequences(test_tokenized, maxlen=MAX_TEXT_LENGTH)


def model_2():
    inp = Input(shape=(MAX_TEXT_LENGTH,))
    model = Embedding(MAX_TEXT_LENGTH,EMBED_SIZE)(inp)
    model = Dropout(0.3)(model)
    model = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(model)
    model = MaxPooling1D(pool_size=4)(model)
    
    model = Dropout(0.25)(model)
    model = LSTM(100)(model)
    model = Dense(10, activation="softmax")(model)
    model = Model(inputs=inp, outputs=model)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def train_fit_predict(model, x_train, x_test, y):
    
    model.fit(x_train, y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS, verbose=1,
              validation_split=VALIDATION_SPLIT)

    return model.predict(x_test)


# Get the list of different classes
CLASSES_LIST = np.unique(y_train)
n_out = len(CLASSES_LIST)
print(CLASSES_LIST)

# Convert class string to index
le = preprocessing.LabelEncoder()
le.fit(CLASSES_LIST)
y_train = le.transform(y_train) 
y_test = le.transform(y_test) 
train_y_cat = np_utils.to_categorical(y_train, n_out)

# get the textual data in the correct format for NN
x_vec_train, x_vec_test = get_train_test(X_train, X_test)
print(len(x_vec_train), len(x_vec_test))

#Normalize
print(x_vec_train.shape)

# define the NN topology
model = model_2()


y_predicted=train_fit_predict(model, x_vec_train, x_vec_test, train_y_cat)


# From proba get a prediction
# We chose the class with the maximum proba
y_pred=[]
for i in range(len(y_test)):
    y_pred.append(np.argmax(y_predicted[i]))
y_pred=np.asarray(y_pred)


print("Test Accuracy: %.2f "% accuracy_score(y_test, y_pred))

p, r, f1, s = precision_recall_fscore_support(y_test, y_pred, 
                                              average='micro',
                                              labels=[x for x in 
                                                      np.unique(y_train) 
                                                      if x not in ['CSDECMOTV']])

print('p r f1:  %.1f %.2f %.3f' % (np.average(p, weights=s)*100.0, 
                                 np.average(r, weights=s)*100.0, 
                                 np.average(f1, weights=s)*100.0))


print(classification_report(y_test, y_pred, labels=[x for x in 
                                                       np.unique(y_train) 
                                                       if x not in ['CSDECMOTV']]))