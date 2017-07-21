# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:39:56 2017

"""
import pandas as pd

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
numberOfWordsToKeep = 15

# create a new dataframe subset to store reviews with total number of words less than or equal to threshold
tempDF = reviewsDF[(reviewsDF.WordCount <= numberOfWordsToKeep)]


#print(reviewsDF)
print(tempDF)
