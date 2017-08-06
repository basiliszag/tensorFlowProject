#Path to Reviews File
path_in = '/GIOTA/DataInput/AmazonReviews/Reviews_new.csv'

#Read file using pandas - creates a data frame
import pandas as pd
import nltk
# create data frame using read_csv from pandas
Reviews = pd.read_csv(path_in, sep=",", header = 'infer')
# keep only score and text columns
Reviews.drop(Reviews.columns[[0,1,2,3,4,5,7,8]], axis=1, inplace=True)

# remove punctuation and then lower case in a new column
Reviews['CleanReview']=Reviews['Text'].astype(str)
Reviews['CleanReview']=Reviews['CleanReview'].str.replace('[^a-zA-Z\s]','').str.lower()


# create one hot vector for score (0-4) 
import numpy as np
def one_hot(i):
    a = np.zeros(5, 'uint8')
    a[i] = 1
    return a

#Convert score to int
Reviews['Score']=Reviews['Score'].astype(int)
#create one hot vector in a separate column
Reviews['Score_label']=Reviews['Score'].apply(lambda x: one_hot(x-1))

#take the length of each review in a new column in order to filter reviews by length
Reviews['WordCount']=Reviews['CleanReview'].apply(len)

#keep reviews with max length 100 words 
Reviews100=Reviews.loc[Reviews['WordCount'] <= 100]

#number of reviews with length 100
#rows=Reviews100.shape[0] #[0] Rows, [1] Columns
#print(rows) #20550 reviews found

#function to remove stop words
def review_words(review_row):
    words = review_row.split()
    from nltk.corpus import stopwords
    meaningful_words = [w for w in words if not w in stopwords.words('english')]
    return( " ".join( meaningful_words ))

#Pre-process the review text and store in a separate column
Reviews100['Review_Words']=Reviews100['CleanReview'].apply(lambda x: review_words(x))

#Add start and end token to final reviews
start_token = "START_T"
end_token = "END_T"
Reviews100['Review_Words']= ["%s %s %s" % (start_token, x, end_token) for x in Reviews100['Review_Words']]

Reviews100.groupby('Score').count()
#1 	1365 
#2 	687 
#3 	1171 
#4 	2750 
#5 	14577 

#we will keep 500 reviews of each score for training and 100 for testing
# Randomly sample 600 reviews of each score as final dataset
Reviews5 = Reviews100.loc[Reviews100['Score'] == 5].sample(600)
Reviews4 = Reviews100.loc[Reviews100['Score'] == 4].sample(600)
Reviews3 = Reviews100.loc[Reviews100['Score'] == 3].sample(600)
Reviews2 = Reviews100.loc[Reviews100['Score'] == 2].sample(600)
Reviews1 = Reviews100.loc[Reviews100['Score'] == 1].sample(600)
ReviewsF = Reviews1.append([Reviews2, Reviews3, Reviews4, Reviews5])

#Join all the words in final review to build a corpus
all_text = ' '.join(ReviewsF['Review_Words'])
words = all_text.split()
# Count the word frequencies
word_freq = nltk.FreqDist(words)
print ("Found %d unique words tokens." % len(word_freq.items())) #Found 3903 unique words tokens.

# Convert words to integers
from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

# Randomly sample 500 reviews of each score for training
Reviews5 = ReviewsF.loc[ReviewsF['Score'] == 5].sample(500)
Reviews4 = ReviewsF.loc[ReviewsF['Score'] == 4].sample(500)
Reviews3 = ReviewsF.loc[ReviewsF['Score'] == 3].sample(500)
Reviews2 = ReviewsF.loc[ReviewsF['Score'] == 2].sample(500)
Reviews1 = ReviewsF.loc[ReviewsF['Score'] == 1].sample(500)
ReviewsTR = Reviews1.append([Reviews2, Reviews3, Reviews4, Reviews5])

# Randomly sample 100 reviews of each score for training
Reviews5 = ReviewsF.loc[ReviewsF['Score'] == 5].sample(100)
Reviews4 = ReviewsF.loc[ReviewsF['Score'] == 4].sample(100)
Reviews3 = ReviewsF.loc[ReviewsF['Score'] == 3].sample(100)
Reviews2 = ReviewsF.loc[ReviewsF['Score'] == 2].sample(100)
Reviews1 = ReviewsF.loc[ReviewsF['Score'] == 1].sample(100)
ReviewsTE = Reviews1.append([Reviews2, Reviews3, Reviews4, Reviews5])

#create reviews to int using the vocabulary for both training and testing dataset
reviews_ints_tr = []
for each in ReviewsTR['Review_Words']:
    reviews_ints_tr.append([vocab_to_int[word] for word in each.split()])

reviews_ints_te = []
for each in ReviewsTE['Review_Words']:
    reviews_ints_te.append([vocab_to_int[word] for word in each.split()])

#take score labels for both training and testing dataset     
score_labels_tr = ReviewsTR['Score_label']
score_labels_te = ReviewsTE['Score_label']

