####Amazon Reviews - Test Version#####


#####################################################################################################
#####################   Prepare Data set to feed the model      #####################################
#####################################################################################################


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

#we will keep 10 reviews of each score for training and 5 for testing
# Randomly sample 15 reviews of each score as final dataset
Reviews5 = Reviews100.loc[Reviews100['Score'] == 5].sample(15)
Reviews4 = Reviews100.loc[Reviews100['Score'] == 4].sample(15)
Reviews3 = Reviews100.loc[Reviews100['Score'] == 3].sample(15)
Reviews2 = Reviews100.loc[Reviews100['Score'] == 2].sample(15)
Reviews1 = Reviews100.loc[Reviews100['Score'] == 1].sample(15)
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
#add empty text as 0
vocab_to_int['EMPTY_T'] = 0

# Randomly sample 10 reviews of each score for training
Reviews5 = ReviewsF.loc[ReviewsF['Score'] == 5].sample(10)
Reviews4 = ReviewsF.loc[ReviewsF['Score'] == 4].sample(10)
Reviews3 = ReviewsF.loc[ReviewsF['Score'] == 3].sample(10)
Reviews2 = ReviewsF.loc[ReviewsF['Score'] == 2].sample(10)
Reviews1 = ReviewsF.loc[ReviewsF['Score'] == 1].sample(10)
ReviewsTR = Reviews1.append([Reviews2, Reviews3, Reviews4, Reviews5])

# Randomly sample 5 reviews of each score for training
Reviews5 = ReviewsF.loc[ReviewsF['Score'] == 5].sample(5)
Reviews4 = ReviewsF.loc[ReviewsF['Score'] == 4].sample(5)
Reviews3 = ReviewsF.loc[ReviewsF['Score'] == 3].sample(5)
Reviews2 = ReviewsF.loc[ReviewsF['Score'] == 2].sample(5)
Reviews1 = ReviewsF.loc[ReviewsF['Score'] == 1].sample(5)
ReviewsTE = Reviews1.append([Reviews2, Reviews3, Reviews4, Reviews5])

#create reviews to int using the vocabulary for both training and testing dataset
reviews_ints_tr = []
for each in ReviewsTR['Review_Words']:
    reviews_ints_tr.append([vocab_to_int[word] for word in each.split()])

reviews_ints_te = []
for each in ReviewsTE['Review_Words']:
    reviews_ints_te.append([vocab_to_int[word] for word in each.split()])

#fill review vectors with 0(-empty text) in order to have fixed length for all vectors
#we use the maximum length of each dataset instead of vocabulary length
def same_len_vector(reviews_int):
    #get maxlen of reviews
    maxlen=0
    for ii, x in enumerate(reviews_int):
        if len(x)>maxlen:
            maxlen=len(x)

    #fill with 0 
    #import numpy as np
    x = np.zeros((len(reviews_int),maxlen), 'uint64')
    for ii, r in enumerate(reviews_int):
        for col in range(0,maxlen-1): 
            if col<len(r):
                x[ii,col]=r[col]
    return x


X_tr=same_len_vector(reviews_ints_tr)
X_te=same_len_vector(reviews_ints_te)

#take score labels for both training and testing dataset     
score_labels_tr = ReviewsTR['Score_label']
score_labels_te = ReviewsTE['Score_label']

#Convert to array in order to feed the model
Y_tr = np.array(score_labels_tr)
Y_tr = np.vstack([np.expand_dims(y, 0) for y in Y_tr])

Y_te = np.array(score_labels_te)
Y_te = np.vstack([np.expand_dims(y, 0) for y in Y_te])

#####################################################################################################
#####################   Build the graph                         #####################################
#####################################################################################################

import tensorflow as tf

#Build the graph
#Define Hyper-Parameters
#lstm_size -> Number of units in the hidden layers in the LSTM cells
#lstm_layers -> Number of LSTM layers in the network. Start with 1, then add more if underfitting
#batch_size -> The number of reviews to feed the network in one training pass. Typically this should be set as high as you can go without running out of memory.
#learning_rate -> Learning rate

lstm_size = 256
lstm_layers = 1
batch_size = 2
learning_rate = 0.001
#Create input placeholders
n_words = len(vocab_to_int)
# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# Embedding - Efficient way to process the input vector is to do embedding instead of one-hot encoding
# Size of the embedding vectors (number of units in the embedding layer)
embed_size = 10
 
with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)
#Build the LSTM cells
with graph.as_default():
    # The basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
 
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
 
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
 
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)
#RNN Forward pass
with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
    initial_state=initial_state)
### Output - Final output of the RNN layer will be used for sentiment prediction.
### So we need to grab the last output with `outputs[:, -1]`, the cost from that and `labels_`.
with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)
 
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
## Graph for checking Validation accuracy
with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
### Batching - Pick only full batches of data and return based on the batch_size
### Version 1 - using yield instead of return - in memory###
def get_batches(x, y, batch_size):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

### Version 2 ###
def get_batches(X, Y, b_size):
    ret = []
    total_batches = X.shape[0]//b_size
    for i in range(total_batches):
        ret.append((X[ i: i+b_size ] , Y[ i: i+b_size ]))
    return ret

#####################################################################################################
#####################   Generic Functions                       #####################################
#####################################################################################################
import sys
def print_same_line(item):
    print(item)
    sys.stdout.write("\033[F")# Cursor up one line

#####################################################################################################
#####################   Training and validation in batches      #####################################
#####################################################################################################

###Training and validation in batches
###Once the graph is defined, training can be done in batches based on the batch_size hyper parameter.
###Models trains to improve the accuracy of the prediction.
batches = get_batches(X_tr, Y_tr, batch_size)
epochs = 10
 
with graph.as_default():
    saver = tf.train.Saver()
 
with tf.Session(graph=graph) as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    #iteration = 1
    iterations = []
    tr_acc = []
    test_acc = []
    costs = []
    for e in range(epochs):
        average_cost = 0.
        average_acc = 0.
        b = 0
        state = sess.run(initial_state)
 
        for ii, (x, y) in enumerate(batches, 1):
            feed = {inputs_: x,
                #labels_: y[:, None],
                labels_: y,
                keep_prob: 0.5,
                initial_state: state}
            batch_acc, loss, state, _ = sess.run([accuracy, cost, final_state, optimizer], feed_dict=feed)
            
            b+=1
            #print_same_line( "Iter " + str(e+1) + " Batch " + str(b) + "/" + str(len(batches)) + " cost " + str(loss) + " tr_acc " + str(batch_acc))
            average_cost += loss/len(batches)
            average_acc += batch_acc/len(batches)
        print( "Iter " + str(e) + ", Minibatch Loss= " + "{:.6f}".format(average_cost)+ ", Minibatch Accuracy= " + "{:.6f}".format(average_acc) )
        # add results to matrices
        iterations.append(e+1)
        tr_acc.append(average_acc)
        #test_acc.append(tes_acc)
        costs.append(average_cost)
        
    print("Optimization Finished!")


#####################################################################################################
##################### Charts for learning curve and train cost  #####################################
#####################################################################################################

%matplotlib inline
%pylab inline
pylab.rcParams['figure.figsize'] = (20, 6)
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

fig1 = plt.figure()

ax = fig1.add_subplot(121)
ax.clear()

ax.set_title("Learning curve")
ax.set_xlabel('Iterations')
ax.set_ylabel('Accuracy')
ax.set_ylim([0, 1])
ax.plot(iterations, tr_acc,     'o-', color="g", label="Train Accuracy")
#ax.plot(iterations, test_acc,   'o-', color="r", label="Test Accuracy")

ax2 = fig1.add_subplot(122)
ax2.clear()

ax2.set_title("Train cost")
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost')
#ax2.set_ylim(bottom=0 )
ax2.set_ylim(ymin=0)
ax2.plot(iterations, costs, 'o-', color="r", label="Train cost")
