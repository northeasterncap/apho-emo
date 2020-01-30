# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:20:19 2020

@author: kryptonh
"""

import pandas as pd
from flask import Flask, jsonify, request
import pickle
import nltk
import re
import string

################################

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()


## Utility functions 
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

def feature_engineer(data):
    data['body_len'] = data['content'].apply(lambda x: len(x) - x.count(" "))
    data['punct%'] = data['content'].apply(lambda x: count_punct(x))
    return data

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text



# load model
model = pickle.load(open('emomodel.pkl','rb'))
transformer = pickle.load(open('emotransformer.pkl','rb'))

# app
app2 = Flask(__name__)

# routes
@app2.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)
  
        # convert data into dataframe
    data_df = pd.DataFrame.from_dict(data,orient = 'columns')
    
    data_df = feature_engineer(data_df)
    
    ###Vectorize
    tfidf_test=transformer.transform(data_df['content'])
    
    X_test_vect = pd.concat([data_df[['body_len', 'punct%']].reset_index(drop=True), 
    pd.DataFrame(tfidf_test.toarray())], axis=1)   
    
     # predictions
    result = model.predict(X_test_vect)
    
    # return data
    return jsonify(result.tolist())

if __name__ == '__main__':
    app2.run(port = 5001, debug=False)