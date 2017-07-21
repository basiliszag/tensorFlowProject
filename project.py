# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:39:56 2017

@author: zagklaras
"""
import pandas as pd

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


# create data frame using read_csv from pandas
reviewsDF = pd.read_csv("C:\\Users\\zagklaras\\Desktop\\Reviews_5k.txt", sep="ω", header = 'infer')

# keep only score and text columns
reviewsDF.drop(reviewsDF.columns[[0,1,2,3,4,5,7,8]], axis=1, inplace=True)

# get names of columns
# list(reviewsDF.columns.values)


# remove punctuation using str.replace from Pandas and regex , and then lower case
# reviewsDF['Text']=reviewsDF['Text'].str.replace('[^\w\s]','').str.lower()
 
reviewsDF['Text']=reviewsDF['Text'].str.replace('[^a-zA-Z\s]','').str.lower()


# create a function that counts total words in a sentence (not unique words)
def wordCount (text):
    length = len(text.split())
    return length

# create a 3rd column in dataframe that holds the total number of words per review
reviewsDF['WordCount'] = reviewsDF.apply(lambda row: wordCount(row['Text']), axis=1)


# threshold  (number of words) for keeping reviews
numberOfWordsToKeep = 50

# create a new dataframe subset to store reviews with total number of words less than or equal to threshold
tempDF = reviewsDF[(reviewsDF.WordCount <= numberOfWordsToKeep)]


#print(reviewsDF)
print(tempDF)



# create bag of stemmed words
# source: https://machinelearnings.co/text-classification-using-neural-networks-f5cd7b8765c6

# initialize words list
words = []
for pattern in tempDF.Text.tolist():
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern)
    # add to our words list
    words.extend(w)
    
    
# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words]
words = list(set(words))

print (len(words), "unique stemmed words", words)
    
