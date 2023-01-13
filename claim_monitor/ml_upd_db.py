'''
CCM
The code below is a derivative of Cards:
"Computer-assisted detection and classification of misinformation about climate change" 
by Travis G. Coan, Constantine Boussalis, John Cook, and Mirjam Nanko.
Cards is licensed under the Apache License 2.0
For information on usage and redistribution, see <https://github.com/traviscoan/cards>
'''

# ML CLASSIFIER AND DB UPDATER
# ----------------------------------------------------------------------
import subprocess
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import ml_utils as u
import pickle
from pymongo import MongoClient
import db as d

nTweets = 400 # Limit for the number of collected tweets

# Import model
# ------------------------------------------------------------------------------
with open("data/model.pkl", 'rb') as f:
    logit = pickle.load(f)

# Define tools
vectorizer = logit['vectorizer']
clf = logit['clf']
le = logit['label_encoder']
clf_logit = clf

def testText(text):
    docs_new = []
    docs_new.append(text)
    X_new_tfidf = vectorizer.transform(docs_new)
    predicted = clf_logit.predict(X_new_tfidf)
    if (predicted == 0):
        print ("category : ", "0 → none")
    elif (predicted == 1):
        print("category : ", "1 → ice isn't melting")
    elif (predicted == 2):
        print("category : ", "2 → we are heading into ice age")
    elif (predicted == 3):
        print("category : ", "3 → weather is cold")
    elif (predicted == 4):
        print("category : ", "4 → there is a hiatus in warning")
    elif (predicted == 5):
        print("category : ", "5 → sea level rise is exaggerated")
    elif (predicted == 6):
        print("category : ", "6 → extremes aren't increasing")
    elif (predicted == 7):
        print("category : ", "7 → it's natural cycles")
    elif (predicted == 8):
        print("category : ", "8 → there is no evidence for greenhouse effect")
    elif (predicted == 9):
        print("category : ", "9 → sensitivity is low")
    elif (predicted == 10):
        print("category : ", "10 → there is no species impact")
    elif (predicted == 11):
        print("category : ", "11 → CO2 is not a pollutant")
    elif (predicted == 12):
        print("category : ", "12 → policies are harmful")
    elif (predicted == 13):
        print("category : ", "13 → policies are ineffective")
    elif (predicted == 14):
        print("category : ", "14 → clean energy won't work")
    elif (predicted == 15):
        print("category : ", "15 → we need energy")
    elif (predicted == 16):
        print("category : ", "16 → science in unreliable")
    elif (predicted == 17):
        print("category : ", "17 → climate movement is unreliable")
    else:
        print("an error occured : no category has been found")