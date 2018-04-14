#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:24:47 2018

@author: liginsolamen
"""
#need gensim library installed 
#pip install --upgrade gensim

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from gensim.models import Word2Vec
import scipy
import scipy.linalg as la
import scipy.misc

def tokenizer(input_str_list,option):
    
#from sklearn.feature_extraction.text import TfidfVectorizer

    if option==1 :
        print('Using TfidfVectorizer')
        n_features=1000
        
        my_vectorizor=TfidfVectorizer(max_df=0.95, min_df=1, max_features=n_features,       stop_words='english', analyzer="word", decode_error="ignore")
        
        
        tfid_transform=my_vectorizor.fit_transform(input_str_list)
        #tfid_feature_name=my_vectorizor.get_feature_names()
    elif option ==2: 
        print('Using Tokenizer NLTK Tokenize & Word Vec')
        
        senttoken=sent_tokenize(input_str_list[0])
        word_token=[]
        for ii in range(1,len(senttoken)):
            word_token.append(word_tokenize(senttoken[ii-1]))
        tfid_transform = Word2Vec(word_token, min_count=1)
        print(tfid_transform)

    return tfid_transform#, tfid_feature_name
